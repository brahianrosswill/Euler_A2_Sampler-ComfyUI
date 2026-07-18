"""Euler-A2 custom sampler node for ComfyUI.

An Euler-ancestral variant: at every step it draws N independent noise paths,
merges them into one stabilized noise direction (mean or per-pixel median),
optionally re-normalizes its magnitude, and extrapolates along that direction.

On top of that it supports:

- Higher-order integration of the deterministic down-step (the probability-flow
  ODE  dx/dσ = (x - denoised)/σ):  euler (1st order), midpoint / heun
  (2nd order), rk4 / ralston4 (4th order), bs3 (3rd order). Euler reproduces
  the classic formula exactly.
- Internal substepping of every sigma interval, in two modes:
    * "ancestral"     - full down-step + renoise per substep (a finer SDE
                        discretization; with matching sigmas this is identical
                        to running on a finer schedule);
    * "deterministic" - substeps only refine the ODE down-step, the renoise
                        happens once per outer step.
  Substep sigmas can be spaced "log" (geometric, recommended) or "linear".
  `substep_stop_at` lets you deactivate substeps after a fraction of the total
  steps, so you can concentrate compute where it matters most (early coarse
  steps) and let late refinement steps run at 1 substep.

Averaging N paths suppresses the per-step noise variance (~1/N for a mean),
which reduces random drift and yields smoother, more coherent ancestral
sampling; `extrapolation` then re-amplifies the merged direction to restore
(or deliberately over/under-shoot) the original noise energy.

Backward compatibility: with the default widget values (method="euler",
substeps=1, substep_stop_at=1.0, noise_paths=2, merge_mode="mean",
noise_normalization="none", extrapolation=0.425, active range [0, 1])
this reproduces the original Euler-A2 behaviour, and old workflows load
unchanged since all new widgets are appended after the original ones.
"""

import math

import torch

import comfy.samplers
from comfy.k_diffusion import sampling as k_diffusion_sampling
from comfy.k_diffusion.sampling import default_noise_sampler
from tqdm.auto import trange

SAMPLER_NAME = "euler_a2"

MERGE_MODES = ("mean", "median")
NORMALIZE_MODES = ("none", "variance", "rms")
METHODS = ("euler", "midpoint", "heun", "rk4", "ralston4", "bs3")
SUBSTEP_MODES = ("ancestral", "deterministic")
SUBSTEP_SPACINGS = ("log", "linear")

_EPS = 1e-8


# ---------------------------------------------------------------------------
# noise helpers
# ---------------------------------------------------------------------------

def _merge_noise_paths(noises, mode):
    """Combine N noise tensors into a single stabilized direction."""
    if len(noises) == 1:
        return noises[0]
    stacked = torch.stack(noises, dim=0)
    if mode == "median":
        # Per-pixel median is robust to outlier draws. Most useful with an
        # odd count >= 3 (torch.median takes the lower middle for even N).
        return stacked.median(dim=0).values
    return stacked.mean(dim=0)


def _normalize_noise(noise, mode, count):
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
        return noise / rms.clamp_min(_EPS)
    return noise


# ---------------------------------------------------------------------------
# integration helpers (probability-flow ODE  dx/dσ = (x - denoised)/σ)
# ---------------------------------------------------------------------------

def _subdivide_sigmas(sigma_a, sigma_b, count, spacing):
    """Return `count + 1` monotone sigma points from sigma_a down to sigma_b (inclusive)."""
    if count <= 1 or sigma_a == sigma_b:
        return [sigma_a, sigma_b]
    if spacing == "log" and sigma_b > 0.0:
        lo, hi = math.log(sigma_a), math.log(sigma_b)
        points = [math.exp(lo + (hi - lo) * j / count) for j in range(count + 1)]
    else:  # uniform in sigma (also the fallback when sigma_b == 0)
        points = [sigma_a + (sigma_b - sigma_a) * j / count for j in range(count + 1)]
    points[0], points[-1] = sigma_a, sigma_b  # exact endpoints
    return points


def _ode_step(model, x, denoised_start, sigma_from, sigma_to, s_in, extra_args, method):
    """Integrate the probability-flow ODE one segment, sigma_from -> sigma_to.

    denoised_start is the cached model prediction at (x, sigma_from).
    Returns (x_new, denoised_end) where denoised_end is the prediction at the
    end point for methods that evaluate it (heun/rk4/ralston4/bs3), else None
    — callers may reuse it as the next segment's denoised_start to save one
    model call. Falls back to Euler when the derivative at sigma_to is
    unavailable (sigma_to == 0) or the segment has zero width.
    """
    if method == "euler" or sigma_to <= 0.0 or sigma_to == sigma_from:
        r = sigma_to / sigma_from
        return r * x + (1.0 - r) * denoised_start, None

    d1 = (x - denoised_start) / sigma_from

    if method == "midpoint":
        sigma_mid = 0.5 * (sigma_from + sigma_to)
        x_mid = x + (sigma_mid - sigma_from) * d1
        denoised_mid = model(x_mid, sigma_mid * s_in, **extra_args)
        d2 = (x_mid - denoised_mid) / sigma_mid
        return x + (sigma_to - sigma_from) * d2, None

    if method == "heun":
        x_end = x + (sigma_to - sigma_from) * d1
        denoised_end = model(x_end, sigma_to * s_in, **extra_args)
        d2 = (x_end - denoised_end) / sigma_to
        return x + 0.5 * (sigma_to - sigma_from) * (d1 + d2), denoised_end

    if method == "rk4":
        # Classical 4th-order Runge-Kutta
        sigma_mid = 0.5 * (sigma_from + sigma_to)
        x_2 = x + (sigma_mid - sigma_from) * d1
        d2 = (x_2 - model(x_2, sigma_mid * s_in, **extra_args)) / sigma_mid
        x_3 = x + (sigma_mid - sigma_from) * d2
        d3 = (x_3 - model(x_3, sigma_mid * s_in, **extra_args)) / sigma_mid
        x_4 = x + (sigma_to - sigma_from) * d3
        denoised_end = model(x_4, sigma_to * s_in, **extra_args)
        d4 = (x_4 - denoised_end) / sigma_to
        return x + (sigma_to - sigma_from) * (d1 + 2.0 * d2 + 2.0 * d3 + d4) / 6.0, denoised_end

    if method == "ralston4":
        # Ralston's 4th-order method — optimized coefficients that minimize
        # the truncation error constant (Taylor error term of order h^5).
        # 4 model evals, same cost as classical RK4 but tighter accuracy.
        h = sigma_to - sigma_from
        k1 = d1
        x2 = x + (2.0 / 5.0) * h * k1
        s2 = sigma_from + (2.0 / 5.0) * h
        k2 = (x2 - model(x2, s2 * s_in, **extra_args)) / s2
        x3 = x + (14.0 / 45.0) * h * k1 + (32.0 / 45.0) * h * k2
        s3 = sigma_from + (14.0 / 45.0) * h
        k3 = (x3 - model(x3, s3 * s_in, **extra_args)) / s3
        x4 = x + (361.0 / 1080.0) * h * k1 + (1184.0 / 1080.0) * h * k2 + (-128.0 / 1080.0) * h * k3
        denoised_end = model(x4, sigma_to * s_in, **extra_args)
        k4 = (x4 - denoised_end) / sigma_to
        return x + h * (0.17476028 * k1 - 0.55148066 * k2 + 1.20553560 * k3 + 0.17118478 * k4), denoised_end

    # bs3 — Bogacki-Shampine 3rd-order method (4 evals, FSAL-capable).
    # Good balance of accuracy and stability for moderate stiffness.
    if method == "bs3":
        h = sigma_to - sigma_from
        k1 = d1
        x2 = x + 0.5 * h * k1
        s2 = sigma_from + 0.5 * h
        k2 = (x2 - model(x2, s2 * s_in, **extra_args)) / s2
        x3 = x + 0.75 * h * k2
        s3 = sigma_from + 0.75 * h
        k3 = (x3 - model(x3, s3 * s_in, **extra_args)) / s3
        x4 = x + (2.0 / 9.0) * h * k1 + (1.0 / 3.0) * h * k2 + (4.0 / 9.0) * h * k3
        denoised_end = model(x4, sigma_to * s_in, **extra_args)
        k4 = (x4 - denoised_end) / sigma_to
        # 3rd-order result (the 4th eval provides FSAL for adaptive stepping,
        # but we use the 3rd-order embedded formula for the output)
        return x + (2.0 / 9.0) * h * k1 + (1.0 / 3.0) * h * k2 + (4.0 / 9.0) * h * k3, denoised_end

    # Fallback (should never happen with validated method strings)
    r = sigma_to / sigma_from
    return r * x + (1.0 - r) * denoised_start, None


def _integrate(model, x, sigma_from, sigma_to, denoised_start, s_in, extra_args,
               method, substeps, spacing, stop_substep_at=1.0):
    """Integrate the ODE from sigma_from to sigma_to, optionally in substeps.

    stop_substep_at: progress fraction (0-1) after which substeps collapse to 1.
    For _integrate, 'progress' is inferred from how far sigma_from sits between
    the outer step's start and end — but since _integrate doesn't know the outer
    loop context, the caller passes the already-resolved effective substep count.
    This function always uses the substeps value as given; the stop_at logic
    lives in the caller (sample_euler_a2) which computes the effective count
    before calling _integrate.
    """
    points = _subdivide_sigmas(sigma_from, sigma_to, substeps, spacing)
    denoised_cached = denoised_start
    for j in range(len(points) - 1):
        x, denoised_end = _ode_step(model, x, denoised_cached, points[j], points[j + 1],
                                    s_in, extra_args, method)
        if j < len(points) - 2:
            denoised_cached = denoised_end if denoised_end is not None \
                else model(x, points[j + 1] * s_in, **extra_args)
    return x


def _ancestral_segment(model, x, sigma_a, sigma_b, denoised_a, *, s_in, extra_args,
                       noise_sampler, eta, s_noise, boost, method, noise_paths,
                       merge_mode, noise_normalization, enhanced_active):
    """One ancestral segment sigma_a -> sigma_b (sigma_b > 0).

    denoised_a is the cached model prediction at (x, sigma_a).
    Returns (x_new, denoised_end) with the same caching convention as _ode_step.
    """
    downstep_ratio = 1.0 + (sigma_b / sigma_a - 1.0) * eta
    sigma_down = min(max(sigma_b * downstep_ratio, 0.0), sigma_a)
    x_det, denoised_end = _ode_step(model, x, denoised_a, sigma_a, sigma_down,
                                    s_in, extra_args, method)

    if eta <= 0.0 or s_noise == 0.0:
        return x_det, denoised_end

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
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    noise_sampler=None,
    eta=1.0,
    s_noise=1.0,
    extrapolation=0.425,
    noise_paths=2,
    merge_mode="mean",
    noise_normalization="none",
    active_start=0.0,
    active_end=1.0,
    method="euler",
    substeps=1,
    substep_stop_at=1.0,
    substep_mode="ancestral",
    substep_spacing="log",
):
    """Euler ancestral sampler merging N noise paths, extrapolated along their shared direction.

    eta:                 ancestral interpolation (0 = deterministic DDIM-like, 1 = full ancestral)
    s_noise:             global multiplier on the injected noise
    extrapolation:       extra gain along the merged direction (total gain = 1 + extrapolation)
    noise_paths:         number of independent noise draws merged per step
    merge_mode:          "mean" or per-pixel "median"
    noise_normalization: "none" | "variance" (x sqrt(N)) | "rms" (unit RMS per sample)
    active_start/end:    fraction of the step range where merging + extrapolation applies
    method:              integration order of the deterministic down-step:
                         "euler" (1 eval), "midpoint"/"heun" (2 evals),
                         "bs3" (3rd order, 4 evals),
                         "rk4"/"ralston4" (4th order, 4 evals) per substep
    substeps:            internal substeps per sigma interval (multiplies model evals)
    substep_stop_at:     fraction of total steps (0-1) after which substeps collapse to 1.
                         e.g. 0.6 = substeps active for first 60% of steps, then off.
                         1.0 = substeps active everywhere (default, backward compatible).
    substep_mode:        "ancestral" (renoise each substep) | "deterministic" (renoise once)
    substep_spacing:     "log" (geometric sigmas) | "linear" (uniform sigmas)
    """
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    noise_paths = max(1, int(noise_paths))
    substeps = max(1, int(substeps))
    substep_stop_at = max(0.0, min(1.0, float(substep_stop_at)))
    boost = 1.0 + extrapolation
    total = len(sigmas) - 1
    denom = max(total - 1, 1)  # maps step index to progress in [0, 1]
    enhanced = noise_paths > 1 or noise_normalization != "none" or extrapolation != 0.0

    # Pre-compute the step index at which substeps deactivate.
    # substep_stop_at=0.6 means: steps with progress < 0.6 use substeps; at >= 0.6 they don't.
    stop_step = int(substep_stop_at * total) if substep_stop_at < 1.0 else total

    for i in trange(total, disable=disable):
        sigma_i = sigmas[i]
        sigma_ip1 = sigmas[i + 1]
        sigma_i_f = float(sigma_i)
        sigma_ip1_f = float(sigma_ip1)

        denoised = model(x, sigma_i * s_in, **extra_args)

        # Degenerate step: nothing left to integrate, the model output is the result.
        if sigma_i_f <= 0.0 or sigma_ip1_f <= 0.0:
            if callback is not None:
                callback({"x": x, "i": i, "sigma": sigma_i, "sigma_hat": sigma_i, "denoised": denoised})
            x = denoised
            continue

        # ---- ancestral down-step (eta interpolates deterministic <-> fully stochastic)
        downstep_ratio = 1.0 + (sigma_ip1_f / sigma_i_f - 1.0) * eta
        sigma_down_f = min(max(sigma_ip1_f * downstep_ratio, 0.0), sigma_i_f)

        if callback is not None:
            sigma_hat = sigma_ip1.new_tensor(sigma_down_f)
            callback({"x": x, "i": i, "sigma": sigma_i, "sigma_hat": sigma_hat, "denoised": denoised})

        progress = i / denom
        enhanced_active = enhanced and active_start <= progress <= active_end
        deterministic = eta <= 0.0 or s_noise == 0.0

        # Resolve the effective substep count for this step.
        eff_substeps = substeps if i < stop_step else 1

        # ---- mode A: finer internal SDE discretization, full ancestral logic per substep
        if not deterministic and substep_mode == "ancestral" and eff_substeps > 1:
            points = _subdivide_sigmas(sigma_i_f, sigma_ip1_f, eff_substeps, substep_spacing)
            denoised_cached = denoised
            for j in range(eff_substeps):
                x, denoised_end = _ancestral_segment(
                    model, x, points[j], points[j + 1], denoised_cached,
                    s_in=s_in, extra_args=extra_args, noise_sampler=noise_sampler,
                    eta=eta, s_noise=s_noise, boost=boost, method=method,
                    noise_paths=noise_paths, merge_mode=merge_mode,
                    noise_normalization=noise_normalization,
                    enhanced_active=enhanced_active)
                if j < eff_substeps - 1:
                    denoised_cached = denoised_end if denoised_end is not None \
                        else model(x, points[j + 1] * s_in, **extra_args)
            continue

        # ---- mode B: integrate the down-step (optionally subdivided), renoise once
        down_substeps = eff_substeps if (substep_mode == "deterministic" or deterministic) else 1
        x_det = _integrate(model, x, sigma_i_f, sigma_down_f, denoised, s_in, extra_args,
                           method, down_substeps, substep_spacing)

        if deterministic:
            x = x_det
            continue

        # ---- renoise up to sigma_{i+1} (alpha = 1 - sigma parametrization)
        alpha_ip1 = 1.0 - sigma_ip1_f
        alpha_down = max(1.0 - sigma_down_f, _EPS)
        base = (alpha_ip1 / alpha_down) * x_det
        renoise_sq = sigma_ip1_f ** 2 - sigma_down_f ** 2 * (alpha_ip1 / alpha_down) ** 2
        noise_scale = s_noise * (max(renoise_sq, 0.0) ** 0.5)

        if noise_scale <= 0.0:
            x = base
            continue

        if enhanced_active:
            noises = [noise_sampler(sigma_i, sigma_ip1) for _ in range(noise_paths)]
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
                "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": False,
                                  "tooltip": "Ancestral interpolation: 0 = deterministic, 1 = full ancestral noise."}),
                "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": False,
                                      "tooltip": "Global multiplier on the injected noise."}),
                "extrapolation": ("FLOAT", {"default": 0.425, "min": -10.0, "max": 10.0, "step": 0.001, "round": False,
                                            "tooltip": "Extra gain along the merged direction (total gain = 1 + this). "
                                                       "~0.414 (sqrt(2)-1) preserves noise energy for 2 paths without normalization."}),
            },
            "optional": {
                "noise_paths": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1,
                                        "tooltip": "Number of independent noise draws merged per step. More paths = smoother, more stable direction."}),
                "merge_mode": (list(MERGE_MODES), {"default": "mean",
                                                   "tooltip": "How to combine the noise paths. 'median' is robust to outlier draws; use 3+ paths."}),
                "noise_normalization": (list(NORMALIZE_MODES), {"default": "none",
                                                                "tooltip": "'variance' rescales by sqrt(N); 'rms' forces unit noise energy every step. "
                                                                           "With either, extrapolation = 0 already preserves energy."}),
                "active_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "round": False,
                                           "tooltip": "Fraction of sampling where path merging begins (0 = first step)."}),
                "active_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "round": False,
                                         "tooltip": "Fraction of sampling where path merging ends (1 = last step). "
                                                    "Restrict the range to keep fine-grained ancestral detail in late steps."}),
                "method": (list(METHODS), {"default": "euler",
                                           "tooltip": "Integration order of the deterministic down-step: "
                                                      "euler (1st, 1 eval), midpoint / heun (2nd, 2 evals), "
                                                      "bs3 (3rd order Bogacki-Shampine, 4 evals), "
                                                      "rk4 (classical 4th, 4 evals), "
                                                      "ralston4 (Ralston 4th, 4 evals — tighter error constant)."}),
                "substeps": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1,
                                     "tooltip": "Split every step into N internal substeps (multiplies model evaluations by N)."}),
                "substep_stop_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "round": False,
                                              "tooltip": "Fraction of total steps (0-1) after which substeps deactivate (collapse to 1). "
                                                         "e.g. 0.6 = substeps for the first 60%% of steps, single-step for the rest. "
                                                         "1.0 = substeps everywhere (default). Concentrate compute on early coarse steps."}),
                "substep_mode": (list(SUBSTEP_MODES), {"default": "ancestral",
                                                       "tooltip": "'ancestral': full down-step + renoise per substep (finer SDE). "
                                                                  "'deterministic': substeps only refine the ODE down-step; renoise once per step."}),
                "substep_spacing": (list(SUBSTEP_SPACINGS), {"default": "log",
                                                             "tooltip": "Internal sigma spacing: 'log' = geometric (recommended), 'linear' = uniform."}),
            },
        }

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "sampling/custom_sampling/samplers"
    DESCRIPTION = ("Euler ancestral sampler that merges several noise paths per step and extrapolates "
                   "along their shared direction, with higher-order integration methods (euler, midpoint, "
                   "heun, rk4, ralston4, bs3) and internal substepping with a configurable stop threshold "
                   "for smoother, more accurate ancestral sampling.")

    def get_sampler(self, eta, s_noise, extrapolation, noise_paths=2, merge_mode="mean",
                    noise_normalization="none", active_start=0.0, active_end=1.0,
                    method="euler", substeps=1, substep_stop_at=1.0,
                    substep_mode="ancestral", substep_spacing="log"):
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
                "substep_stop_at": substep_stop_at,
                "substep_mode": substep_mode,
                "substep_spacing": substep_spacing,
            },
        )
        return (sampler,)


NODE_CLASS_MAPPINGS = {
    "Euler_A2_Sampler": EulerA2Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Euler_A2_Sampler": "Euler A2 Sampler",
}
