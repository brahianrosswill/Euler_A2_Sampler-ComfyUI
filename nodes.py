"""Euler-A2 custom sampler node for ComfyUI.

An Euler-ancestral variant: at every step it draws N independent noise paths,
merges them into one stabilized noise direction (mean or per-pixel median),
optionally re-normalizes its magnitude, and extrapolates along that direction.

Averaging N paths suppresses the per-step noise variance (~1/N for a mean),
which reduces random drift and yields smoother, more coherent ancestral
sampling; `extrapolation` then re-amplifies the merged direction to restore
(or deliberately over/under-shoot) the original noise energy.

Backward compatibility: with the default widget values (noise_paths=2,
merge_mode="mean", noise_normalization="none", extrapolation=0.425,
active range [0, 1]) this reproduces the original Euler-A2 behaviour, and
old workflows load unchanged since the first three widgets keep their order.
"""

import torch

import comfy.samplers
from comfy.k_diffusion import sampling as k_diffusion_sampling
from comfy.k_diffusion.sampling import default_noise_sampler
from tqdm.auto import trange

SAMPLER_NAME = "euler_a2"

MERGE_MODES = ("mean", "median")
NORMALIZE_MODES = ("none", "variance", "rms")

_EPS = 1e-8


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
):
    """Euler ancestral sampler merging N noise paths, extrapolated along their shared direction.

    eta:                 ancestral interpolation (0 = deterministic DDIM-like, 1 = full ancestral)
    s_noise:             global multiplier on the injected noise
    extrapolation:       extra gain along the merged direction (total gain = 1 + extrapolation)
    noise_paths:         number of independent noise draws merged per step
    merge_mode:          "mean" or per-pixel "median"
    noise_normalization: "none" | "variance" (x sqrt(N)) | "rms" (unit RMS per sample)
    active_start/end:    fraction of the step range where merging + extrapolation applies;
                         outside it a plain single-noise ancestral step is used
    """
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    noise_paths = max(1, int(noise_paths))
    boost = 1.0 + extrapolation
    total = len(sigmas) - 1
    denom = max(total - 1, 1)  # maps step index to progress in [0, 1]
    enhanced = noise_paths > 1 or noise_normalization != "none" or extrapolation != 0.0

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
        sigma_down_i_ratio = sigma_down_f / sigma_i_f
        deterministic_path = sigma_down_i_ratio * x + (1.0 - sigma_down_i_ratio) * denoised

        if callback is not None:
            sigma_hat = sigma_ip1.new_tensor(sigma_down_f)
            callback({"x": x, "i": i, "sigma": sigma_i, "sigma_hat": sigma_hat, "denoised": denoised})

        if eta <= 0.0 or s_noise == 0.0:
            x = deterministic_path
            continue

        # ---- renoise up to sigma_{i+1} (alpha = 1 - sigma parametrization)
        alpha_ip1 = 1.0 - sigma_ip1_f
        alpha_down = max(1.0 - sigma_down_f, _EPS)
        base = (alpha_ip1 / alpha_down) * deterministic_path
        renoise_sq = sigma_ip1_f ** 2 - sigma_down_f ** 2 * (alpha_ip1 / alpha_down) ** 2
        noise_scale = s_noise * (max(renoise_sq, 0.0) ** 0.5)

        if noise_scale <= 0.0:
            x = base
            continue

        progress = i / denom
        if enhanced and active_start <= progress <= active_end:
            noises = [noise_sampler(sigma_i, sigma_ip1) for _ in range(noise_paths)]
            direction = _merge_noise_paths(noises, merge_mode)
            direction = _normalize_noise(direction, noise_normalization, noise_paths)
            x = base + direction * (noise_scale * boost)
        else:
            x = base + noise_sampler(sigma_i, sigma_ip1) * noise_scale

    return x


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
            },
        }

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "sampling/custom_sampling/samplers"
    DESCRIPTION = ("Euler ancestral sampler that merges several noise paths per step and extrapolates "
                   "along their shared direction for smoother, more stable ancestral sampling.")

    def get_sampler(self, eta, s_noise, extrapolation, noise_paths=2, merge_mode="mean",
                    noise_normalization="none", active_start=0.0, active_end=1.0):
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
            },
        )
        return (sampler,)


NODE_CLASS_MAPPINGS = {
    "Euler_A2_Sampler": EulerA2Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Euler_A2_Sampler": "Euler A2 Sampler",
}
