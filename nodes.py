"""Euler-A2 custom sampler node for ComfyUI.

An Euler-ancestral variant: at every step it draws N independent noise paths,
merges them into one stabilized noise direction (mean or per-pixel median),
optionally re-normalizes its magnitude, and extrapolates along that direction.

On top of that it supports:

- Higher-order integration of the deterministic down-step (the probability-flow
  ODE  dx/dσ = (x - denoised)/σ):  euler (1st order), midpoint / heun
  (2nd order), rk4 (4th order), ralston3 (3rd order, 3 evals),
  rk4_38 (3/8-rule, 4 evals), ssprk3 (SSP 3rd order, 3 evals),
  dpm_2s (DPM-Solver++ 2S, 2 evals).
  Euler reproduces the classic formula exactly.
- Internal substepping of every sigma interval, in two modes:
    * "ancestral"     - full down-step + renoise per substep (a finer SDE
                        discretization; with matching sigmas this is identical
                        to running on a finer schedule);
    * "deterministic" - substeps only refine the ODE down-step, the renoise
                        happens once per outer step.
  Substep sigmas can be spaced "log" (geometric, recommended) or "linear".
- Substep stop sigmas: comma-separated descending sigma values that split
  intervals so full substeps only apply above each threshold; segments at
  or below the smallest stop sigma use single steps.

Averaging N paths suppresses the per-step noise variance (~1/N for a mean),
which reduces random drift and yields smoother, more coherent ancestral
sampling; `extrapolation` then re-amplifies the merged direction to restore
(or deliberately over/under-shoot) the original noise energy.

Backward compatibility: with the default widget values (method="euler",
substeps=1, noise_paths=2, merge_mode="mean", noise_normalization="none",
extrapolation=0.425, active range [0, 1]) this reproduces the original
Euler-A2 behaviour, and old workflows load unchanged since all new widgets
are appended after the original ones.
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
METHODS = ("euler", "midpoint", "heun", "rk4",
           "ralston3", "rk4_38", "ssprk3", "dpm_2s")
SUBSTEP_MODES = ("ancestral", "deterministic")
SUBSTEP_SPACINGS = ("log", "linear")

_EPS = 1e-8

# ---------------------------------------------------------------------------
# RK tableau lookup  (used by ralston3, rk4_38, ssprk3)
# ---------------------------------------------------------------------------
# Each entry:  (a_coefficients, b_weights, n_stages)
# a is lower-triangular (list of lists, row k has k entries)
# b is the final weight vector;  x_new = x + h * sum(b_i * k_i)

_RK_TABLEAUX = {
    # Ralston's 3rd-order method (3 stages, 3 model evals)
    "ralston3": (
        [[],
         [2/3],
         [0, 1]],
        [1/4, 0, 3/4],
        3,
    ),
    # 3/8-rule RK4 (4 stages, 4 model evals — same order as classical RK4)
    "rk4_38": (
        [[],
         [1/3],
         [-1/3, 1],
         [1, -1, 1]],
        [1/8, 3/8, 3/8, 1/8],
        4,
    ),
    # SSPRK3 — Shu-Osher 3rd-order, 3 stages (strong-stability-preserving)
    "ssprk3": (
        [[],
         [1],
         [1/4, 1/4]],
        [1/6, 1/6, 2/3],
        3,
    ),
}


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


def _subdivide_at_stops(sigma_a, sigma_b, count, spacing, stop_sigmas):
    """Split [sigma_a, sigma_b] at stop_sigmas boundaries.

    Returns a list of (sigma_from, sigma_to, substep_count) tuples.
    Segments whose start is at or below the smallest stop sigma get 1 substep;
    segments above the smallest stop sigma get the full `count` substeps.
    stop_sigmas must be sorted descending.
    """
    boundaries = [sigma_a]
    for s in stop_sigmas:
        if sigma_b < s < sigma_a:
            boundaries.append(s)
    boundaries.append(sigma_b)

    # The smallest stop sigma is the last element (list is descending)
    smallest_stop = stop_sigmas[-1]

    segments = []
    for k in range(len(boundaries) - 1):
        a, b = boundaries[k], boundaries[k + 1]
        # Apply full substeps only if this segment starts above the smallest stop
        use_count = count if a > smallest_stop else 1
        segments.append((a, b, use_count))
    return segments


def _rk_step(tableau, weights, n_stages, model, x, d1,
             sigma_from, sigma_to, s_in, extra_args):
    """Generic explicit Runge-Kutta step using a tableau.

    d1 is the slope k₀ = (x - denoised_start) / sigma_from already evaluated.
    Returns (x_new, None) — no endpoint caching for tableau methods.
    """
    h = sigma_to - sigma_from
    ks = [None] * n_stages
    ks[0] = d1
    for s in range(1, n_stages):
        x_s = x
        for j in range(s):
            x_s = x_s + h * tableau[s][j] * ks[j]
        # sigma at stage s (sum of a-coefficients gives the fractional position)
        sigma_s = sigma_from + h * sum(tableau[s])
        denoised_s = model(x_s, max(sigma_s, 0.0) * s_in, **extra_args)
        ks[s] = (x_s - denoised_s) / max(sigma_s, _EPS)
    return x + h * sum(w * k for w, k in zip(weights, ks)), None


def _ode_step(model, x, denoised_start, sigma_from, sigma_to, s_in, extra_args, method):
    """Integrate the probability-flow ODE one segment, sigma_from -> sigma_to.

    denoised_start is the cached model prediction at (x, sigma_from).
    Returns (x_new, denoised_end) where denoised_end is the prediction at the
    end point for methods that evaluate it (heun/rk4), else None — callers may
    reuse it as the next segment's denoised_start to save one model call.
    Falls back to Euler when the derivative at sigma_to is unavailable
    (sigma_to == 0) or the segment has zero width.
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

    # ---- Tableau-based RK methods ----
    if method in _RK_TABLEAUX:
        tableau, weights, n_stages = _RK_TABLEAUX[method]
        return _rk_step(tableau, weights, n_stages, model, x, d1,
                        sigma_from, sigma_to, s_in, extra_args)

    # rk4 (classical 4th order — kept as its own branch for clarity)
    if method == "rk4":
        sigma_mid = 0.5 * (sigma_from + sigma_to)
        x_2 = x + (sigma_mid - sigma_from) * d1
        d2 = (x_2 - model(x_2, sigma_mid * s_in, **extra_args)) / sigma_mid
        x_3 = x + (sigma_mid - sigma_from) * d2
        d3 = (x_3 - model(x_3, sigma_mid * s_in, **extra_args)) / sigma_mid
        x_4 = x + (sigma_to - sigma_from) * d3
        denoised_end = model(x_4, sigma_to * s_in, **extra_args)
        d4 = (x_4 - denoised_end) / sigma_to
        return x + (sigma_to - sigma_from) * (d1 + 2.0 * d2 + 2.0 * d3 + d4) / 6.0, denoised_end

    # DPM-Solver++ 2S  (exponential half-step solver for semi-linear ODE)
    #   Uses the exact solution of dx/dσ = (x - denoised)/σ in log-σ space.
    #   2 model evals per step, 2nd order, tuned for diffusion sampling.
    if method == "dpm_2s":
        if sigma_to <= 0.0:
            r = sigma_to / sigma_from
            return r * x + (1.0 - r) * denoised_start, None
        lambda_from = math.log(sigma_from)
        lambda_to = math.log(sigma_to)
        h = lambda_to - lambda_from
        lambda_mid = lambda_from + 0.5 * h
        sigma_mid = math.exp(lambda_mid)
        # half-step with Euler (exponential integrator)
        exp_h2 = math.exp(0.5 * h)
        x_mid = exp_h2 * x + (1.0 - exp_h2) * denoised_start
        denoised_mid = model(x_mid, sigma_mid * s_in, **extra_args)
        # full-step with midpoint correction
        exp_h = math.exp(h)
        x_end = exp_h * x + (1.0 - exp_h) * denoised_mid
        return x_end, denoised_mid

    # Fallback: should not be reached if METHODS is kept in sync
    r = sigma_to / sigma_from
    return r * x + (1.0 - r) * denoised_start, None


def _integrate(model, x, sigma_from, sigma_to, denoised_start, s_in, extra_args,
               method, substeps, spacing, stop_sigmas=None):
    """Integrate the ODE from sigma_from to sigma_to, optionally in substeps.

    stop_sigmas: list of sigma values (descending) at which substeps should
    stop.  Segments at or below the smallest stop sigma use 1 substep;
    segments above get the full `substeps` count.
    """
    if stop_sigmas:
        segments = _subdivide_at_stops(sigma_from, sigma_to, substeps, spacing, stop_sigmas)
    else:
        segments = [(sigma_from, sigma_to, substeps)]

    denoised_cached = denoised_start
    for seg_from, seg_to, seg_count in segments:
        points = _subdivide_sigmas(seg_from, seg_to, seg_count, spacing)
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
    substep_mode="ancestral",
    substep_spacing="log",
    substep_stop_sigmas=None,
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
                         "euler" (1 eval), "midpoint"/"heun" (2 evals), "rk4" (4 evals),
                         "ralston3" (3rd order, 3 evals), "rk4_38" (3/8-rule, 4 evals),
                         "ssprk3" (SSP 3rd order, 3 evals), "dpm_2s" (DPM-Solver++ 2S, 2 evals)
    substeps:            internal substeps per sigma interval (multiplies model evals)
    substep_mode:        "ancestral" (renoise each substep) | "deterministic" (renoise once)
    substep_spacing:     "log" (geometric sigmas) | "linear" (uniform sigmas)
    substep_stop_sigmas: comma-separated descending sigma values where substeps stop;
                         segments at or below the smallest stop sigma use 1 substep
    """
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    noise_paths = max(1, int(noise_paths))
    substeps = max(1, int(substeps))
    boost = 1.0 + extrapolation
    total = len(sigmas) - 1
    denom = max(total - 1, 1)  # maps step index to progress in [0, 1]
    enhanced = noise_paths > 1 or noise_normalization != "none" or extrapolation != 0.0

    # Parse substep stop sigmas (comma-separated descending values)
    if substep_stop_sigmas is not None:
        if isinstance(substep_stop_sigmas, str):
            _stops = [float(s.strip()) for s in substep_stop_sigmas.split(',') if s.strip()]
        else:
            _stops = list(substep_stop_sigmas) if substep_stop_sigmas else []
        substep_stop_sigmas = sorted(set(_stops), reverse=True) if _stops else None
    else:
        substep_stop_sigmas = None

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

        # ---- mode A: finer internal SDE discretization, full ancestral logic per substep
        if not deterministic and substep_mode == "ancestral" and substeps > 1:
            if substep_stop_sigmas:
                segments = _subdivide_at_stops(sigma_i_f, sigma_ip1_f, substeps,
                                               substep_spacing, substep_stop_sigmas)
            else:
                segments = [(sigma_i_f, sigma_ip1_f, substeps)]
            denoised_cached = denoised
            for seg_from, seg_to, seg_count in segments:
                points = _subdivide_sigmas(seg_from, seg_to, seg_count, substep_spacing)
                for j in range(seg_count):
                    x, denoised_end = _ancestral_segment(
                        model, x, points[j], points[j + 1], denoised_cached,
                        s_in=s_in, extra_args=extra_args, noise_sampler=noise_sampler,
                        eta=eta, s_noise=s_noise, boost=boost, method=method,
                        noise_paths=noise_paths, merge_mode=merge_mode,
                        noise_normalization=noise_normalization,
                        enhanced_active=enhanced_active)
                    if j < seg_count - 1:
                        denoised_cached = denoised_end if denoised_end is not None \
                            else model(x, points[j + 1] * s_in, **extra_args)
            continue

        # ---- mode B: integrate the down-step (optionally subdivided), renoise once
        down_substeps = substeps if (substep_mode == "deterministic" or deterministic) else 1
        x_det = _integrate(model, x, sigma_i_f, sigma_down_f, denoised, s_in, extra_args,
                           method, down_substeps, substep_spacing,
                           stop_sigmas=substep_stop_sigmas)

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
                                                      "euler (1st order, 1 eval), midpoint / heun (2nd order, 2 evals), "
                                                      "rk4 (4th order, 4 evals), ralston3 (3rd order, 3 evals), "
                                                      "rk4_38 (3/8-rule, 4 evals), ssprk3 (SSP 3rd order, 3 evals), "
                                                      "dpm_2s (DPM-Solver++ 2S, 2 evals) per substep."}),
                "substeps": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1,
                                     "tooltip": "Split every step into N internal substeps (multiplies model evaluations by N)."}),
                "substep_mode": (list(SUBSTEP_MODES), {"default": "ancestral",
                                                       "tooltip": "'ancestral': full down-step + renoise per substep (finer SDE). "
                                                                  "'deterministic': substeps only refine the ODE down-step; renoise once per step."}),
                "substep_spacing": (list(SUBSTEP_SPACINGS), {"default": "log",
                                                             "tooltip": "Internal sigma spacing: 'log' = geometric (recommended), 'linear' = uniform."}),
                "substep_stop_sigmas": ("STRING", {"default": "",
                                                    "tooltip": "Comma-separated descending sigma values where substeps stop. "
                                                               "Segments at or below the smallest stop sigma use 1 substep. "
                                                               "Example: '5.0,2.0,0.5' applies full substeps only above sigma 5.0, "
                                                               "between 5.0-2.0, and between 2.0-0.5; below 0.5 uses single steps."}),
            },
        }

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "sampling/custom_sampling/samplers"
    DESCRIPTION = ("Euler ancestral sampler that merges several noise paths per step and extrapolates "
                   "along their shared direction, with higher-order integration methods (euler, midpoint, "
                   "heun, rk4, ralston3, rk4_38, ssprk3, dpm_2s) and internal substepping with optional "
                   "stop-sigma boundaries for smoother, more accurate ancestral sampling.")

    def get_sampler(self, eta, s_noise, extrapolation, noise_paths=2, merge_mode="mean",
                    noise_normalization="none", active_start=0.0, active_end=1.0,
                    method="euler", substeps=1, substep_mode="ancestral", substep_spacing="log",
                    substep_stop_sigmas=""):
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
                "substep_stop_sigmas": substep_stop_sigmas,
            },
        )
        return (sampler,)


NODE_CLASS_MAPPINGS = {
    "Euler_A2_Sampler": EulerA2Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Euler_A2_Sampler": "Euler A2 Sampler",
}
