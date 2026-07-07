import comfy.samplers
import torch
import torch.nn.functional as F
from comfy.k_diffusion import sampling as k_diffusion_sampling
from comfy.k_diffusion.sampling import default_noise_sampler
from tqdm.auto import trange


SAMPLER_NAME = "euler_a2_v2"


def _append_unique(target, value):
    if value not in target:
        target.append(value)


@torch.no_grad()
def sample_euler_a2_v2(
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
    merge_mode="average",
    noise_correlation=0.0,
    momentum=0.0,
    normalize_direction=False,
    adaptive_scale=False,
    seed=-1,
    step_start=0,
    step_end=10000,
    clamp_denoised=False,
    **kwargs,
):
    """
    Improved Euler Ancestral A2 sampler with dual-path noise extrapolation.
    
    Generates two parallel noise paths, merges them via configurable strategy,
    and extrapolates along the merged direction with optional momentum and
    adaptive scaling.
    """
    extra_args = {} if extra_args is None else extra_args
    
    # Seed resolution: explicit param > extra_args > random
    actual_seed = seed if seed != -1 else extra_args.get("seed", None)
    if noise_sampler is None:
        noise_sampler = default_noise_sampler(x, seed=actual_seed)
    
    s_in = x.new_ones([x.shape[0]])
    prev_direction = None
    total_steps = len(sigmas) - 1
    
    for i in trange(total_steps, disable=disable):
        # Guard against zero sigma to prevent division by zero
        if sigmas[i] == 0:
            x = model(x, sigmas[i] * s_in, **extra_args)
            if callback is not None:
                callback({
                    "x": x, "i": i, "sigma": sigmas[i],
                    "sigma_hat": sigmas[i], "denoised": x
                })
            continue
        
        # Model prediction
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        
        if clamp_denoised:
            denoised = torch.clamp(denoised, -1.0, 1.0)
        
        if callback is not None:
            callback({
                "x": x, "i": i, "sigma": sigmas[i],
                "sigma_hat": sigmas[i], "denoised": denoised
            })
        
        # Final step: return clean prediction
        if sigmas[i + 1] == 0:
            x = denoised
            continue
        
        # Out-of-range steps: standard Euler (no ancestral noise)
        if i < step_start or i >= step_end:
            d = (x - denoised) / sigmas[i].clamp_min(1e-8)
            dt = sigmas[i + 1] - sigmas[i]
            x = x + d * dt
            prev_direction = None
            continue
        
        # Ancestral downstep ratio
        downstep_ratio = 1 + (sigmas[i + 1] / sigmas[i] - 1) * eta
        sigma_down = (sigmas[i + 1] * downstep_ratio).clamp_min(0.0)
        
        # Alpha values with numerical stability clamp
        alpha_ip1 = (1 - sigmas[i + 1]).clamp_min(1e-6)
        alpha_down = (1 - sigma_down).clamp_min(1e-6)
        
        # Deterministic ancestral path
        sigma_down_i_ratio = sigma_down / sigmas[i]
        deterministic_path = sigma_down_i_ratio * x + (1 - sigma_down_i_ratio) * denoised
        
        if eta > 0 and s_noise != 0:
            # Base for noise addition
            base = (alpha_ip1 / alpha_down) * deterministic_path
            
            # Renoise coefficient with stability guard
            renoise_term = (
                sigmas[i + 1] ** 2 - sigma_down ** 2 * alpha_ip1 ** 2 / alpha_down ** 2
            )
            renoise_coeff = renoise_term.clamp_min(0).sqrt()
            noise_scale = s_noise * renoise_coeff
            
            # Generate two independent noise samples
            noise_1 = noise_sampler(sigmas[i], sigmas[i + 1])
            noise_2 = noise_sampler(sigmas[i], sigmas[i + 1])
            
            # Apply correlation between noise paths
            if abs(noise_correlation) > 1e-6:
                noise_2 = (
                    noise_correlation * noise_1
                    + (1 - noise_correlation ** 2).sqrt() * noise_2
                )
            
            # Construct dual paths
            path_1 = base + noise_1 * noise_scale
            path_2 = base + noise_2 * noise_scale
            
            # Merge strategies
            if merge_mode == "average":
                merged = 0.5 * (path_1 + path_2)
            elif merge_mode == "weighted":
                # Inverse variance weighting
                dist_1 = (path_1 - deterministic_path).abs().mean()
                dist_2 = (path_2 - deterministic_path).abs().mean()
                total_dist = dist_1 + dist_2 + 1e-8
                w1 = dist_2 / total_dist
                w2 = dist_1 / total_dist
                merged = w1 * path_1 + w2 * path_2
            elif merge_mode == "min":
                merged = torch.minimum(path_1, path_2)
            elif merge_mode == "max":
                merged = torch.maximum(path_1, path_2)
            elif merge_mode == "difference":
                merged = base + 0.5 * (path_1 - path_2)
            else:
                merged = 0.5 * (path_1 + path_2)
            
            # Extrapolation direction
            direction = merged - base
            
            # Optional direction normalization
            if normalize_direction:
                dir_norm = direction.norm(
                    dim=tuple(range(1, direction.ndim)), keepdim=True
                )
                direction = direction / (dir_norm + 1e-8)
                merged_norm = merged.norm(
                    dim=tuple(range(1, merged.ndim)), keepdim=True
                )
                direction = direction * merged_norm
            
            # Momentum: blend with previous direction
            if momentum > 0 and prev_direction is not None:
                direction = momentum * prev_direction + (1 - momentum) * direction
            
            # Adaptive scale: reduce extrapolation at high sigma (early steps)
            if adaptive_scale and sigmas[0] > 0:
                sigma_ratio = sigmas[i] / sigmas[0]  # 1.0 → 0.0
                adaptive_extr = extrapolation * (0.5 + 0.5 * (1 - sigma_ratio))
            else:
                adaptive_extr = extrapolation
            
            # Store direction for next step momentum
            prev_direction = direction.clone()
            x = merged + adaptive_extr * direction
        else:
            # Deterministic only (eta=0 or s_noise=0)
            x = deterministic_path
            prev_direction = None
    
    return x


def _register_sampler():
    """Idempotent sampler registration."""
    func_name = f"sample_{SAMPLER_NAME}"
    if not hasattr(k_diffusion_sampling, func_name):
        setattr(k_diffusion_sampling, func_name, sample_euler_a2_v2)
        _append_unique(comfy.samplers.KSAMPLER_NAMES, SAMPLER_NAME)
        _append_unique(comfy.samplers.SAMPLER_NAMES, SAMPLER_NAME)
        _append_unique(comfy.samplers.KSampler.SAMPLERS, SAMPLER_NAME)


_register_sampler()


class EulerA2Sampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "eta": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 100.0,
                    "step": 0.01, "round": False
                }),
                "s_noise": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 100.0,
                    "step": 0.01, "round": False
                }),
                "extrapolation": ("FLOAT", {
                    "default": 0.425, "min": -10.0, "max": 10.0,
                    "step": 0.001, "round": False
                }),
            },
            "optional": {
                "merge_mode": (["average", "weighted", "min", "max", "difference"], {
                    "default": "average"
                }),
                "noise_correlation": ("FLOAT", {
                    "default": 0.0, "min": -1.0, "max": 1.0,
                    "step": 0.01, "round": False
                }),
                "momentum": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.99,
                    "step": 0.01, "round": False
                }),
                "normalize_direction": ("BOOLEAN", {"default": False}),
                "adaptive_scale": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {
                    "default": -1, "min": -1, "max": 40000000000,
                }),
                "step_start": ("INT", {
                    "default": 0, "min": 0, "max": 10000
                }),
                "step_end": ("INT", {
                    "default": 10000, "min": 0, "max": 10000
                }),
                "clamp_denoised": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "get_sampler"
    CATEGORY = "sampling/custom_sampling/samplers"
    DESCRIPTION = (
        "Improved Euler Ancestral A2 with dual-path extrapolation, "
        "momentum, noise correlation, and step-range control."
    )

    @classmethod
    def VALIDATE_INPUTS(cls, merge_mode, step_start, step_end, **kwargs):
        if step_start > step_end:
            return "step_start must be <= step_end"
        return True

    def get_sampler(
        self,
        eta,
        s_noise,
        extrapolation,
        merge_mode="average",
        noise_correlation=0.0,
        momentum=0.0,
        normalize_direction=False,
        adaptive_scale=False,
        seed=-1,
        step_start=0,
        step_end=10000,
        clamp_denoised=False,
    ):
        sampler = comfy.samplers.ksampler(
            SAMPLER_NAME,
            {
                "eta": eta,
                "s_noise": s_noise,
                "extrapolation": extrapolation,
                "merge_mode": merge_mode,
                "noise_correlation": noise_correlation,
                "momentum": momentum,
                "normalize_direction": normalize_direction,
                "adaptive_scale": adaptive_scale,
                "seed": seed,
                "step_start": step_start,
                "step_end": step_end,
                "clamp_denoised": clamp_denoised,
            },
        )
        return (sampler,)


NODE_CLASS_MAPPINGS = {
    "Euler_A2_Sampler": EulerA2Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Euler_A2_Sampler": "Euler A2 Sampler (Improved)",
}
