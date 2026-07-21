"""Euler-A2 custom sampler node for ComfyUI.

An Euler-ancestral variant: at every step it draws N independent noise paths,
merges them into one stabilized noise direction (mean or per-pixel median),
optionally re-normalizes its magnitude, and extrapolates along that direction.

On top of that it supports:

- Higher-order integration of the deterministic down-step (the probability-flow
  ODE  dx/dσ = (x - denoised)/σ):  euler (1st order), midpoint / ralston /
  heun / dpm2 (2nd order), rk3 (3rd order), rk4 (4th order), and **ab2**
  (Adams-Bashforth 2-step: 2nd order, 1 evaluation per step after the first,
  using the previous step's derivative for a multistep predictor).
- Internal substepping of every sigma interval, in two modes:
    * "ancestral"     - full down-step + renoise per substep (a finer SDE
                        discretization; with matching sigmas this is identical
                        to running on a finer schedule);
    * "deterministic" - substeps only refine the ODE down-step, the renoise
                        happens once per outer step.
  Substep sigmas can be spaced "log" (geometric, recommended) or "linear".
- **Practical substep scheduling**: substeps can be restricted to a fraction
  of the sampling process (`substep_active_start` / `substep_active_end`) and
  faded in/out (`substep_fade`) so model evaluations aren't wasted on steps
  where they add little value.
- **Parameterization support**: "flow" (α = 1 − σ, original Euler-A2 behaviour)
  or "edm" (α = 1, standard k-diffusion ancestral sampling compatible with
  most SD 1.5 / SDXL checkpoints).

Averaging N paths suppresses the per-step noise variance (~1/N for a mean),
which reduces random drift and yields smoother, more coherent ancestral
sampling; `extrapolation` then re-amplifies the merged direction to restore
(or deliberately over/under-shoot) the original noise energy.

Backward compatibility: with the default widget values (method="euler",
substeps=1, substep_active_start=0.0, substep_active_end=1.0, substep_fade=0.0,
noise_paths=2, merge_mode="mean", noise_normalization="none",
extrapolation=0.425, active range [0, 1], parameterization="flow") this
reproduces the original Euler-A2 behaviour, and old workflows load unchanged
since all new widgets are appended after the original ones.
"""

import math
from typing import List, Tuple, Optional

import torch
from torch import Tensor

import comfy.samplers
from comfy.k_diffusion import sampling as k_diffusion_sampling
from comfy.k_diffusion.sampling import default_noise_sampler
from tqdm.auto import trange

SAMPLER_NAME = "euler_a2"

MERGE_MODES = ("mean", "median")
NORMALIZE_MODES = ("none", "variance", "rms")
METHODS = ("euler", "midpoint", "ralston", "heun", "dpm2", "rk3", "rk4", "ab2")
SUBSTEP_MODES = ("ancestral", "deterministic")
SUBSTEP_SPACINGS = ("log", "linear")
PARAMETERIZATIONS = ("flow", "edm")

_EPS = 1e-8


# ---------------------------------------------------------------------------
# noise helpers
# ---------------------------------------------------------------------------

def _merge_noise_paths(noises: List[Tensor], mode: str) -> Tensor:
    """Combine N noise tensors into a single stabilized direction."""
    if len(noises) == 1:
        return noises[0]
    stacked = torch.stack(noises, dim=0)
    if mode == "median":
        # Per-pixel median is robust to outlier draws. Most useful with an
        # odd count >= 3 (torch.median takes the lower middle for even N).
        return stacked.median(dim=0).values
    return stacked.mean(dim=0)


def _normalize_noise(noise: Tensor, mode: str, count: int) -> Tensor:
    """Rescale a merged noise direction.

    none     - keep as-is (a mean of N draws has variance ~1/N)
    variance - statistical correction: multiply by sqrt(N) to restore unit variance
    rms      - empirical correction: force per-sample RMS to exactly 1, which
               removes energy fluctuations between steps and seeds
    """
    if mode == "variance":
        return noise * (float(count) ** 0.5)
    if mode == "rms":
        if noise.ndim > 1:
            dims = tuple(range(1, noise.ndim))
            rms = noise.pow(2).mean(dim=dims, keepdim=True).sqrt()
        else:
            rms = noise.pow(2).mean().sqrt()
        # Guard against all-zero latents (would blow up to 1/_EPS).
        if rms.max() < _EPS:
            return noise
        return noise / rms.clamp_min(_EPS)
    return noise


# ---------------------------------------------------------------------------
# substep scheduling helper
# ---------------------------------------------------------------------------

def _effective_substeps(
    substeps: int,
    progress: float,
    active_start: float,
    active_end: float,
    fade: float,
) -> int:
    """Return the number of internal substeps for a given outer-step progress.

    progress       – i / (total_steps - 1), i.e. 0 at first step, 1 at last.
    active_start   – fraction where substeps begin.
    active_end     – fraction where substeps end.
    fade           – fraction of the active window used to ramp substeps
                     from 1 up to N (at the start) and back down (at the end).
                     Values > 0.5 are clamped to 0.5 to prevent overlapping
                     ramp-up / ramp-down windows.
    """
    if substeps <= 1:
        return 1
    if progress < active_start or progress > active_end:
        return 1
    if fade <= 0.0:
        return substeps
    active_range = active_end - active_start
    if active_range <= 0.0:
        return substeps
    # Clamp fade to prevent overlapping ramp-up/ramp-down windows.
    fade = min(fade, 0.5)
    fade_width = fade * active_range
    if progress < active_start + fade_width:
        t = (progress - active_start) / fade_width
        return max(1, int(round(1.0 + (substeps - 1) * t)))
    if progress > active_end - fade_width:
        t = (active_end - progress) / fade_width
        return max(1, int(round(1.0 + (substeps - 1) * t)))
    return substeps


# ---------------------------------------------------------------------------
# integration helpers (probability-flow ODE  dx/dσ = (x - denoised)/σ)
# ---------------------------------------------------------------------------

def _subdivide_sigmas(
    sigma_a: float, sigma_b: float, count: int, spacing: str
) -> List[float]:
    """Return `count + 1` monotone sigma points from sigma_a down to sigma_b (inclusive)."""
    if count <= 1 or abs(sigma_a - sigma_b) < _EPS:
        return [sigma_a, sigma_b]
    if spacing == "log" and sigma_b > 0.0:
        lo, hi = math.log(sigma_a), math.log(sigma_b)
        points = [math.exp(lo + (hi - lo) * j / count) for j in range(count + 1)]
    else:
        # Uniform in sigma (also the fallback when sigma_b == 0).
        points = [sigma_a + (sigma_b - sigma_a) * j / count for j in range(count + 1)]
    points[0], points[-1] = sigma_a, sigma_b  # exact endpoints
    return points


def _ode_step(
    model,
    x: Tensor,
    denoised_start: Tensor,
    sigma_from: float,
    sigma_to: float,
    s_in: Tensor,
    extra_args,
    method: str,
    prev_derivative: Optional[Tensor] = None,
    prev_sigma: Optional[float] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """Integrate the probability-flow ODE one segment, sigma_from -> sigma_to.

    denoised_start is the cached model prediction at (x, sigma_from).
    prev_derivative and prev_sigma are the derivative (x - denoised)/sigma and
    sigma from the previous evaluation point, used by multistep methods (ab2).
    Returns (x_new, denoised_end) where denoised_end is the prediction at the
    end point for methods that evaluate it (heun/rk4/rk3), else None — callers
    may reuse it as the next segment's denoised_start to save one model call.
    Falls back to Euler when the derivative at sigma_to is unavailable
    (sigma_to == 0) or the segment has zero width.
    """
    if method == "euler" or sigma_to <= 0.0 or abs(sigma_to - sigma_from) < _EPS:
        r = sigma_to / sigma_from
        return r * x + (1.0 - r) * denoised_start, None

    d1 = (x - denoised_start) / sigma_from
    h = sigma_to - sigma_from

    if method == "ab2":
        # Adams-Bashforth 2-step (2nd order, 1 evaluation).
        # Variable-step formula derived from linear interpolation of the
        # derivative between (prev_sigma, prev_derivative) and (sigma_from, d1).
        if prev_derivative is None or prev_sigma is None or abs(sigma_from - prev_sigma) < _EPS:
            # Fallback to Euler on first step or when previous data is missing/invalid.
            return x + h * d1, None
        h_prev = sigma_from - prev_sigma
        # Coefficients from integrating the linear interpolant.
        c1 = 1.0 + h / (2.0 * h_prev)
        c2 = h / (2.0 * h_prev)
        return x + h * (c1 * d1 - c2 * prev_derivative), None

    if method == "midpoint":
        sigma_mid = 0.5 * (sigma_from + sigma_to)
        x_mid = x + (sigma_mid - sigma_from) * d1
        denoised_mid = model(x_mid, sigma_mid * s_in, **extra_args)
        d2 = (x_mid - denoised_mid) / sigma_mid
        return x + h * d2, None

    if method == "ralston":
        # Ralston's method (2nd order, 2 evaluations).
        # Minimizes the truncation-error bound compared to the standard midpoint rule.
        sigma_r = sigma_from + (2.0 / 3.0) * h
        x_r = x + (2.0 / 3.0) * h * d1
        denoised_r = model(x_r, sigma_r * s_in, **extra_args)
        d2 = (x_r - denoised_r) / sigma_r
        return x + h * (0.25 * d1 + 0.75 * d2), None

    if method == "heun":
        x_end = x + h * d1
        denoised_end = model(x_end, sigma_to * s_in, **extra_args)
        d2 = (x_end - denoised_end) / sigma_to
        return x + 0.5 * h * (d1 + d2), denoised_end

    if method == "dpm2":
        # DPM-Solver-2 (data-prediction variant, single-step, 2nd order).
        # Uses the geometric midpoint and an exponential-integrator correction
        # tailored to the diffusion probability-flow ODE.
        sigma_mid = (sigma_from * sigma_to) ** 0.5
        r_mid = sigma_mid / sigma_from
        x_mid = r_mid * x + (1.0 - r_mid) * denoised_start
        denoised_mid = model(x_mid, sigma_mid * s_in, **extra_args)
        r = sigma_to / sigma_from
        x_next = (
            r * x
            + (1.0 - r) * denoised_start
            + 0.5 * (1.0 - r) * (denoised_mid - denoised_start)
        )
        return x_next, None

    if method == "rk3":
        # Classical Runge-Kutta 3rd order (Kutta's method, 3 evaluations).
        sigma_mid = 0.5 * (sigma_from + sigma_to)
        x_2 = x + 0.5 * h * d1
        denoised_2 = model(x_2, sigma_mid * s_in, **extra_args)
        d2 = (x_2 - denoised_2) / sigma_mid
        x_3 = x - h * d1 + 2.0 * h * d2
        denoised_3 = model(x_3, sigma_to * s_in, **extra_args)
        d3 = (x_3 - denoised_3) / sigma_to
        return x + h * (d1 + 4.0 * d2 + d3) / 6.0, denoised_3

    if method == "rk4":
        # Classical Runge-Kutta 4th order
        sigma_mid = 0.5 * (sigma_from + sigma_to)
        x_2 = x + (sigma_mid - sigma_from) * d1
        d2 = (x_2 - model(x_2, sigma_mid * s_in, **extra_args)) / sigma_mid
        x_3 = x + (sigma_mid - sigma_from) * d2
        d3 = (x_3 - model(x_3, sigma_mid * s_in, **extra_args)) / sigma_mid
        x_4 = x + h * d3
        denoised_end = model(x_4, sigma_to * s_in, **extra_args)
        d4 = (x_4 - denoised_end) / sigma_to
        return x + h * (d1 + 2.0 * d2 + 2.0 * d3 + d4) / 6.0, denoised_end

    raise ValueError(f"Unknown ODE integration method: {method}")


def _integrate(
    model,
    x: Tensor,
    sigma_from: float,
    sigma_to: float,
    denoised_start: Tensor,
    s_in: Tensor,
    extra_args,
    method: str,
    substeps: int,
    spacing: str,
    prev_derivative: Optional[Tensor] = None,
    prev_sigma: Optional[float] = None,
) -> Tensor:
    """Integrate the ODE from sigma_from to sigma_to, optionally in substeps."""
    points = _subdivide_sigmas(sigma_from, sigma_to, substeps, spacing)
    denoised_cached = denoised_start
    for j in range(len(points) - 1):
        # Only the first substep may use the multistep history;
        # AB2 does not produce an endpoint derivative cheaply, so subsequent
        # substeps automatically fall back to Euler inside _ode_step.
        pd = prev_derivative if j == 0 else None
        ps = prev_sigma if j == 0 else None
        x, denoised_end = _ode_step(
            model, x, denoised_cached, points[j], points[j + 1],
            s_in, extra_args, method, pd, ps
        )
        if j < len(points) - 2:
            denoised_cached = (
                denoised_end
                if denoised_end is not None
                else model(x, points[j + 1] * s_in, **extra_args)
            )
    return x


def _ancestral_segment(
    model,
    x: Tensor,
    sigma_a: float,
    sigma_b: float,
    denoised_a: Tensor,
    *,
    s_in: Tensor,
    extra_args,
    noise_sampler,
    eta: float,
    s_noise: float,
    boost: float,
    method: str,
    noise_paths: int,
    merge_mode: str,
    noise_normalization: str,
    enhanced_active: bool,
    parameterization: str,
    prev_derivative: Optional[Tensor] = None,
    prev_sigma: Optional[float] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """One ancestral segment sigma_a -> sigma_b (sigma_b > 0).

    denoised_a is the cached model prediction at (x, sigma_a).
    prev_derivative and prev_sigma are passed through to _ode_step for
    multistep methods (ab2).
    Returns (x_new, denoised_end) with the same caching convention as _ode_step.
    """
    if parameterization == "edm":
        sigma_down, sigma_up = k_diffusion_sampling.get_ancestral_step(sigma_a, sigma_b, eta=eta)
    else:
        downstep_ratio = 1.0 + (sigma_b / sigma_a - 1.0) * eta
        sigma_down = min(max(sigma_b * downstep_ratio, 0.0), sigma_a)
        sigma_up = None

    x_det, denoised_end = _ode_step(
        model, x, denoised_a, sigma_a, sigma_down, s_in, extra_args, method,
        prev_derivative, prev_sigma
    )

    if eta <= 0.0 or s_noise == 0.0:
        return x_det, denoised_end

    if parameterization == "edm":
        if sigma_up is None:
            sigma_up = math.sqrt(max(sigma_b ** 2 - sigma_down ** 2, 0.0))
        if sigma_up <= 0.0:
            return x_det, denoised_end
        noise_scale = s_noise * sigma_up
        if enhanced_active:
            noises = [noise_sampler(sigma_a, sigma_b) for _ in range(noise_paths)]
            direction = _merge_noise_paths(noises, merge_mode)
            direction = _normalize_noise(direction, noise_normalization, noise_paths)
            return x_det + direction * (noise_scale * boost), denoised_end
        return x_det + noise_sampler(sigma_a, sigma_b) * noise_scale, denoised_end

    # Flow parameterization (original Euler-A2 behaviour)
    alpha_b = 1.0 - sigma_b
    alpha_down = max(1.0 - sigma_down, _EPS)
    base = (alpha_b / alpha_down) * x_det
    renoise_sq = sigma_b ** 2 - sigma_down ** 2 * (alpha_b / alpha_down) ** 2
    noise_scale = s_noise * (max(renoise_sq, 0.0) ** 0.5)
    if noise_scale <= 0.0:
        return base, denoised_end

    if enhanced_active:
        noises = [noise_sampler(sigma_a, sigma_b) for _ in range(noise_paths)]
        direction = _merge_noise_paths(noises, merge_mode)
        direction = _normalize_noise(direction, noise_normalization, noise_paths)
        return base + direction * (noise_scale * boost), denoised_end
    return base + noise_sampler(sigma_a, sigma_b) * noise_scale, denoised_end


# ---------------------------------------------------------------------------
# sampler
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_euler_a2(
    model,
    x: Tensor,
    sigmas: Tensor,
    extra_args=None,
    callback=None,
    disable=None,
    noise_sampler=None,
    eta: float = 1.0,
    s_noise: float = 1.0,
    extrapolation: float = 0.425,
    noise_paths: int = 2,
    merge_mode: str = "mean",
    noise_normalization: str = "none",
    active_start: float = 0.0,
    active_end: float = 1.0,
    method: str = "euler",
    substeps: int = 1,
    substep_mode: str = "ancestral",
    substep_spacing: str = "log",
    substep_active_start: float = 0.0,
    substep_active_end: float = 1.0,
    substep_fade: float = 0.0,
    parameterization: str = "flow",
):
    """Euler ancestral sampler merging N noise paths, extrapolated along their shared direction.

    eta:                 ancestral interpolation (0 = deterministic DDIM-like, 1 = full ancestral)
    s_noise:             global multiplier on the injected noise
    extrapolation:       extra gain along the merged direction (total gain = 1 + this)
    noise_paths:         number of independent noise draws merged per step
    merge_mode:          "mean" or per-pixel "median"
    noise_normalization: "none" | "variance" (x sqrt(N)) | "rms" (unit RMS per sample)
    active_start/end:    fraction of the step range where merging + extrapolation applies
    method:              integration order of the deterministic down-step:
                         euler (1 eval), midpoint / ralston / heun / dpm2 (2 evals),
                         rk3 (3 evals), rk4 (4 evals), ab2 (1 eval, multistep) per substep.
    substeps:            max internal substeps per sigma interval (multiplies model evals)
    substep_mode:        "ancestral" (renoise each substep) | "deterministic" (renoise once)
    substep_spacing:     "log" (geometric sigmas) | "linear" (uniform sigmas)
    substep_active_start/end:
                         fraction of sampling where internal substeps are active.
                         Defaults [0, 1] = always active (backward compatible).
    substep_fade:        fraction of the active window used to ramp substeps from
                         1 up to N and back down. 0 = abrupt on/off.
                         Values > 0.5 are clamped internally to prevent overlap.
    parameterization:    "flow" (α = 1 − σ, original Euler-A2, best for flow-matching
                         models) or "edm" (α = 1, standard k-diffusion ancestral,
                         best for EDM / SD 1.5 / SDXL models).
    """
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = (
        default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    )
    s_in = x.new_ones([x.shape[0]])

    noise_paths = max(1, int(noise_paths))
    substeps = max(1, int(substeps))
    boost = 1.0 + extrapolation
    total = len(sigmas) - 1
    denom = max(total - 1, 1)  # maps step index to progress in [0, 1]
    enhanced = (
        noise_paths > 1 or noise_normalization != "none" or extrapolation != 0.0
    )

    # Multistep state: cache the previous step's derivative for Adams-Bashforth.
    prev_derivative = None
    prev_sigma = None

    for i in trange(total, disable=disable):
        sigma_i = sigmas[i]
        sigma_ip1 = sigmas[i + 1]
        sigma_i_f = float(sigma_i)
        sigma_ip1_f = float(sigma_ip1)

        denoised = model(x, sigma_i * s_in, **extra_args)

        # Degenerate step: nothing left to integrate, the model output is the result.
        if sigma_i_f <= 0.0 or sigma_ip1_f <= 0.0:
            # Reset multistep history across discontinuities (e.g. sigma <= 0).
            prev_derivative = None
            prev_sigma = None
            if callback is not None:
                sigma_hat = sigma_ip1.new_tensor(sigma_i_f)
                callback(
                    {
                        "x": x,
                        "i": i,
                        "sigma": sigma_i,
                        "sigma_hat": sigma_hat,
                        "denoised": denoised,
                    }
                )
            x = denoised
            continue

        # Cache the derivative at the current point for the next step's multistep predictor.
        current_derivative = (x - denoised) / sigma_i_f

        # ---- compute sigma_down based on parameterization
        if parameterization == "edm":
            sigma_down_f, sigma_up = k_diffusion_sampling.get_ancestral_step(
                sigma_i_f, sigma_ip1_f, eta=eta
            )
        else:
            downstep_ratio = 1.0 + (sigma_ip1_f / sigma_i_f - 1.0) * eta
            sigma_down_f = min(max(sigma_ip1_f * downstep_ratio, 0.0), sigma_i_f)
            sigma_up = None

        if callback is not None:
            sigma_hat = sigma_ip1.new_tensor(sigma_down_f)
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigma_i,
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )

        progress = i / denom
        enhanced_active = enhanced and active_start <= progress <= active_end
        deterministic = eta <= 0.0 or s_noise == 0.0

        # ---- compute effective substeps for this outer step (practical scheduling)
        effective_substeps = _effective_substeps(
            substeps, progress, substep_active_start, substep_active_end, substep_fade
        )

        # ---- mode A: finer internal SDE discretization, full ancestral logic per substep
        if (
            not deterministic
            and substep_mode == "ancestral"
            and effective_substeps > 1
        ):
            points = _subdivide_sigmas(
                sigma_i_f, sigma_ip1_f, effective_substeps, substep_spacing
            )
            denoised_cached = denoised
            for j in range(effective_substeps):
                x, denoised_end = _ancestral_segment(
                    model,
                    x,
                    points[j],
                    points[j + 1],
                    denoised_cached,
                    s_in=s_in,
                    extra_args=extra_args,
                    noise_sampler=noise_sampler,
                    eta=eta,
                    s_noise=s_noise,
                    boost=boost,
                    method=method,
                    noise_paths=noise_paths,
                    merge_mode=merge_mode,
                    noise_normalization=noise_normalization,
                    enhanced_active=enhanced_active,
                    parameterization=parameterization,
                    prev_derivative=prev_derivative if j == 0 else None,
                    prev_sigma=prev_sigma if j == 0 else None,
                )
                if j < effective_substeps - 1:
                    denoised_cached = (
                        denoised_end
                        if denoised_end is not None
                        else model(x, points[j + 1] * s_in, **extra_args)
                    )
            # Update multistep history after the outer step completes.
            prev_derivative = current_derivative
            prev_sigma = sigma_i_f
            continue

        # ---- mode B: integrate the down-step (optionally subdivided), renoise once
        down_substeps = (
            effective_substeps
            if (substep_mode == "deterministic" or deterministic)
            else 1
        )
        x_det = _integrate(
            model,
            x,
            sigma_i_f,
            sigma_down_f,
            denoised,
            s_in,
            extra_args,
            method,
            down_substeps,
            substep_spacing,
            prev_derivative,
            prev_sigma,
        )

        # Update multistep history after the deterministic down-step.
        prev_derivative = current_derivative
        prev_sigma = sigma_i_f

        if deterministic:
            x = x_det
            continue

        # ---- renoise up to sigma_{i+1}
        if parameterization == "edm":
            if sigma_up is None:
                sigma_up = math.sqrt(max(sigma_ip1_f ** 2 - sigma_down_f ** 2, 0.0))
            if sigma_up <= 0.0:
                x = x_det
                continue
            noise_scale = s_noise * sigma_up
            if enhanced_active:
                noises = [
                    noise_sampler(sigma_i, sigma_ip1) for _ in range(noise_paths)
                ]
                direction = _merge_noise_paths(noises, merge_mode)
                direction = _normalize_noise(direction, noise_normalization, noise_paths)
                x = x_det + direction * (noise_scale * boost)
            else:
                x = x_det + noise_sampler(sigma_i, sigma_ip1) * noise_scale
        else:
            # Flow parameterization (α = 1 − σ)
            alpha_ip1 = 1.0 - sigma_ip1_f
            alpha_down = max(1.0 - sigma_down_f, _EPS)
            base = (alpha_ip1 / alpha_down) * x_det
            renoise_sq = (
                sigma_ip1_f ** 2
                - sigma_down_f ** 2 * (alpha_ip1 / alpha_down) ** 2
            )
            noise_scale = s_noise * (max(renoise_sq, 0.0) ** 0.5)
            if noise_scale <= 0.0:
                x = base
                continue
            if enhanced_active:
                noises = [
                    noise_sampler(sigma_i, sigma_ip1) for _ in range(noise_paths)
                ]
                direction = _merge_noise_paths(noises, merge_mode)
                direction = _normalize_noise(direction, noise_normalization, noise_paths)
                x = base + direction * (noise_scale * boost)
            else:
                x = base + noise_sampler(sigma_i, sigma_ip1) * noise_scale

    return x


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

def _append_unique(target, value):
    if value not in target:
        target.append(value)


def _register_sampler():
    """Register sample_euler_a2 with ComfyUI's sampler lists (idempotent)."""
    setattr(k_diffusion_sampling, f"sample_{SAMPLER_NAME}", sample_euler_a2)

    registries = [
        getattr(comfy.samplers, "KSAMPLER_NAMES", None),
        getattr(comfy.samplers, "SAMPLER_NAMES", None),
    ]
    ksampler_cls = getattr(comfy.samplers, "KSampler", None)
    if ksampler_cls is not None:
        registries.append(getattr(ksampler_cls, "SAMPLERS", None))

    for registry in registries:
        if isinstance(registry, list):
            _append_unique(registry, SAMPLER_NAME)


_register_sampler()


# ---------------------------------------------------------------------------
# node
# ---------------------------------------------------------------------------

class EulerA2Sampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "eta": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "round": False,
                        "tooltip": "Ancestral interpolation: 0 = deterministic, 1 = full ancestral noise.",
                    },
                ),
                "s_noise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "round": False,
                        "tooltip": "Global multiplier on the injected noise.",
                    },
                ),
                "extrapolation": (
                    "FLOAT",
                    {
                        "default": 0.425,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.001,
                        "round": False,
                        "tooltip": "Extra gain along the merged direction (total gain = 1 + this). "
                        "~0.414 (sqrt(2)-1) preserves noise energy for 2 paths without normalization.",
                    },
                ),
            },
            "optional": {
                "noise_paths": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "tooltip": "Number of independent noise draws merged per step. More paths = smoother, more stable direction.",
                    },
                ),
                "merge_mode": (
                    list(MERGE_MODES),
                    {
                        "default": "mean",
                        "tooltip": "How to combine the noise paths. 'median' is robust to outlier draws; use 3+ paths.",
                    },
                ),
                "noise_normalization": (
                    list(NORMALIZE_MODES),
                    {
                        "default": "none",
                        "tooltip": "'variance' rescales by sqrt(N); 'rms' forces unit noise energy every step. "
                        "With either, extrapolation = 0 already preserves energy.",
                    },
                ),
                "active_start": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "round": False,
                        "tooltip": "Fraction of sampling where path merging begins (0 = first step).",
                    },
                ),
                "active_end": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "round": False,
                        "tooltip": "Fraction of sampling where path merging ends (1 = last step). "
                        "Restrict the range to keep fine-grained ancestral detail in late steps.",
                    },
                ),
                "method": (
                    list(METHODS),
                    {
                        "default": "euler",
                        "tooltip": "Integration order of the deterministic down-step: "
                        "euler (1st order, 1 eval), midpoint / ralston / heun / dpm2 (2nd order, 2 evals), "
                        "rk3 (3rd order, 3 evals), rk4 (4th order, 4 evals), "
                        "ab2 (2nd order multistep, 1 eval, uses previous derivative) per substep.",
                    },
                ),
                "substeps": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "tooltip": "Max internal substeps per sigma interval (multiplies model evaluations by this).",
                    },
                ),
                "substep_mode": (
                    list(SUBSTEP_MODES),
                    {
                        "default": "ancestral",
                        "tooltip": "'ancestral': full down-step + renoise per substep (finer SDE). "
                        "'deterministic': substeps only refine the ODE down-step; renoise once per step.",
                    },
                ),
                "substep_spacing": (
                    list(SUBSTEP_SPACINGS),
                    {
                        "default": "log",
                        "tooltip": "Internal sigma spacing: 'log' = geometric (recommended), 'linear' = uniform.",
                    },
                ),
                "substep_active_start": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "round": False,
                        "tooltip": "Fraction of sampling where internal substeps begin. "
                        "e.g. 0.0 = from the first step, 0.3 = skip the first 30 %.",
                    },
                ),
                "substep_active_end": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "round": False,
                        "tooltip": "Fraction of sampling where internal substeps end. "
                        "e.g. 0.5 = stop after the first half of steps.",
                    },
                ),
                "substep_fade": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "round": False,
                        "tooltip": "Fade substeps in/out over this fraction of the active window. "
                        "0 = abrupt on/off; 0.25 = smooth ramp over 25 % of the active range. "
                        "Values > 0.5 are clamped internally to prevent overlapping ramps.",
                    },
                ),
                "parameterization": (
                    list(PARAMETERIZATIONS),
                    {
                        "default": "flow",
                        "tooltip": "'flow' = α = 1−σ (original Euler-A2, best for flow-matching models). "
                        "'edm' = α = 1 (standard k-diffusion ancestral, best for SD 1.5 / SDXL / EDM).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "sampling/custom_sampling/samplers"
    DESCRIPTION = (
        "Euler ancestral sampler that merges several noise paths per step and extrapolates "
        "along their shared direction, with higher-order integration methods (including "
        "Adams-Bashforth 2-step multistep), internal substepping, practical substep scheduling, "
        "and EDM/flow parameterization support for smoother, more efficient ancestral sampling."
    )

    def get_sampler(
        self,
        eta,
        s_noise,
        extrapolation,
        noise_paths=2,
        merge_mode="mean",
        noise_normalization="none",
        active_start=0.0,
        active_end=1.0,
        method="euler",
        substeps=1,
        substep_mode="ancestral",
        substep_spacing="log",
        substep_active_start=0.0,
        substep_active_end=1.0,
        substep_fade=0.0,
        parameterization="flow",
    ):
        sampler = comfy.samplers.ksampler(
            SAMPLER_NAME,
            {
                "eta": eta,
                "s_noise": s_noise,
                "extrapolation": extrapolation,
                "noise_paths": noise_paths,
                "merge_mode": merge_mode,
                "noise_normalization": noise_normalization,
                "active_start": active_start,
                "active_end": active_end,
                "method": method,
                "substeps": substeps,
                "substep_mode": substep_mode,
                "substep_spacing": substep_spacing,
                "substep_active_start": substep_active_start,
                "substep_active_end": substep_active_end,
                "substep_fade": substep_fade,
                "parameterization": parameterization,
            },
        )
        return (sampler,)


NODE_CLASS_MAPPINGS = {
    "Euler_A2_Sampler": EulerA2Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Euler_A2_Sampler": "Euler A2 Sampler",
}
