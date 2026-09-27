import comfy.samplers
import torch
from comfy.k_diffusion import sampling as k_diffusion_sampling
from comfy.k_diffusion.sampling import default_noise_sampler
from tqdm.auto import trange


SAMPLER_NAME = "euler_a2"

# Available noise normalization modes (used both for the sampler and the node UI).
NOISE_NORM_MODES = [
    "none",
    "l2",
    "rms",
    "unit_var",
    "l1",
    "linf",
    "batch_l2",
    "batch_rms",
    "snr",
]


def _normalize_noise(noise, mode, sigma=None, sigma_next=None, eps=1e-12):
    """Normalize a noise tensor according to `mode`.

    Per-sample modes reduce over all non-batch dimensions so each batch element
    is normalized independently.  Batch modes reduce over everything.
    """
    if noise is None or mode == "none":
        return noise

    dims = tuple(range(1, noise.ndim))  # all dims except batch

    if mode == "l2":
        norm = noise.pow(2).sum(dim=dims, keepdim=True).clamp_min(eps).sqrt()
        return noise / norm

    if mode == "rms":
        norm = noise.pow(2).mean(dim=dims, keepdim=True).clamp_min(eps).sqrt()
        return noise / norm

    if mode == "unit_var":
        mean = noise.mean(dim=dims, keepdim=True)
        var = (noise - mean).pow(2).mean(dim=dims, keepdim=True).clamp_min(eps)
        return (noise - mean) / var.sqrt()

    if mode == "l1":
        norm = noise.abs().mean(dim=dims, keepdim=True).clamp_min(eps)
        return noise / norm

    if mode == "linf":
        norm = noise.abs().amax(dim=dims, keepdim=True).clamp_min(eps)
        return noise / norm

    if mode == "batch_l2":
        norm = noise.pow(2).sum().clamp_min(eps).sqrt()
        return noise / norm

    if mode == "batch_rms":
        norm = noise.pow(2).mean().clamp_min(eps).sqrt()
        return noise / norm

    if mode == "snr":
        # Rescale noise to track the sigma schedule.  At sigmas[i] the noise
        # injected back into the sample should scale roughly with sigmas[i+1].
        if sigma is None or sigma_next is None:
            return noise
        ratio = (sigma_next / sigma).clamp_min(eps)
        # Normalize to RMS=1 first, then apply the schedule ratio.
        norm = noise.pow(2).mean(dim=dims, keepdim=True).clamp_min(eps).sqrt()
        return (noise / norm) * ratio

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
    noise_norm="none",
):
    """Euler ancestral sampler that averages two noise paths and extrapolates along their mean direction."""
    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    noise_sampler = default_noise_sampler(x, seed=seed) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})

        if sigmas[i + 1] == 0:
            x = denoised
            continue

        downstep_ratio = 1 + (sigmas[i + 1] / sigmas[i] - 1) * eta
        sigma_down = sigmas[i + 1] * downstep_ratio
        alpha_ip1 = 1 - sigmas[i + 1]
        alpha_down = 1 - sigma_down

        sigma_down_i_ratio = sigma_down / sigmas[i]
        deterministic_path = sigma_down_i_ratio * x + (1 - sigma_down_i_ratio) * denoised

        if eta > 0 and s_noise != 0:
            base = (alpha_ip1 / alpha_down) * deterministic_path
            renoise_coeff = (
                sigmas[i + 1] ** 2 - sigma_down ** 2 * alpha_ip1 ** 2 / alpha_down ** 2
            ).clamp_min(0).sqrt()
            noise_scale = s_noise * renoise_coeff

            raw_1 = noise_sampler(sigmas[i], sigmas[i + 1])
            raw_2 = noise_sampler(sigmas[i], sigmas[i + 1])

            noise_1 = _normalize_noise(
                raw_1, noise_norm, sigma=sigmas[i], sigma_next=sigmas[i + 1]
            )
            noise_2 = _normalize_noise(
                raw_2, noise_norm, sigma=sigmas[i], sigma_next=sigmas[i + 1]
            )

            path_1 = base + noise_1 * noise_scale
            path_2 = base + noise_2 * noise_scale
            merged = 0.5 * (path_1 + path_2)
            direction = merged - base
            x = merged + extrapolation * direction
        else:
            x = deterministic_path

    return x


def _append_unique(target, value):
    if value not in target:
        target.append(value)


def _register_sampler():
    setattr(k_diffusion_sampling, f"sample_{SAMPLER_NAME}", sample_euler_a2)

    _append_unique(comfy.samplers.KSAMPLER_NAMES, SAMPLER_NAME)
    _append_unique(comfy.samplers.SAMPLER_NAMES, SAMPLER_NAME)
    _append_unique(comfy.samplers.KSampler.SAMPLERS, SAMPLER_NAME)


_register_sampler()


class EulerA2Sampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": False}),
                "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": False}),
                "extrapolation": ("FLOAT", {"default": 0.425, "min": -10.0, "max": 10.0, "step": 0.001, "round": False}),
                "noise_norm": (NOISE_NORM_MODES, {"default": "none"}),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "sampling/custom_sampling/samplers"

    def get_sampler(self, eta, s_noise, extrapolation, noise_norm="none"):
        sampler = comfy.samplers.ksampler(
            SAMPLER_NAME,
            {
                "eta": eta,
                "s_noise": s_noise,
                "extrapolation": extrapolation,
                "noise_norm": noise_norm,
            },
        )
        return (sampler,)


NODE_CLASS_MAPPINGS = {
    "Euler_A2_Sampler": EulerA2Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Euler_A2_Sampler": "Euler_A2_Sampler",
}