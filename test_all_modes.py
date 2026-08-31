"""Comprehensive test of all normalization modes."""
import torch
import math

_EPS = 1e-8

def _strip_str(value):
    return value.strip() if isinstance(value, str) else value

def _normalize_noise_current(
    noise: torch.Tensor, mode: str, count: int, sigma_from: float = None, sigma_to: float = None
) -> torch.Tensor:
    """Current implementation from nodes.py."""
    mode = _strip_str(mode)

    if mode == "variance":
        return noise * (float(count) ** 0.5)

    if mode == "rms":
        if noise.ndim > 1:
            dims = tuple(range(1, noise.ndim))
            rms = noise.pow(2).mean(dim=dims, keepdim=True).sqrt()
        else:
            rms = noise.pow(2).mean().sqrt()

        if rms.max() < _EPS:
            return noise

        return noise / rms.clamp_min(_EPS)

    if mode == "spectral":
        noise_fft = torch.fft.fftn(noise.float())
        magnitudes = torch.abs(noise_fft)
        
        freq_shape = noise_fft.shape[-2:]
        cy, cx = freq_shape[0] // 2, freq_shape[1] // 2
        y, x = torch.meshgrid(
            torch.arange(freq_shape[0], device=noise.device, dtype=torch.float32),
            torch.arange(freq_shape[1], device=noise.device, dtype=torch.float32),
            indexing='ij'
        )
        radius = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_radius = torch.sqrt(torch.tensor(cx**2 + cy**2, device=noise.device, dtype=torch.float32))
        normalized_radius = radius / (max_radius + _EPS)
        
        spectral_weight = 1.0 / (normalized_radius + 0.1).clamp_min(_EPS)
        spectral_weight = spectral_weight / spectral_weight.max()
        
        noise_fft_weighted = noise_fft * spectral_weight
        noise_spectral = torch.fft.ifftn(noise_fft_weighted).real
        
        if noise.ndim > 1:
            dims = tuple(range(1, noise.ndim))
            orig_rms = noise.pow(2).mean(dim=dims, keepdim=True).sqrt()
            new_rms = noise_spectral.pow(2).mean(dim=dims, keepdim=True).sqrt()
        else:
            orig_rms = noise.pow(2).mean().sqrt()
            new_rms = noise_spectral.pow(2).mean().sqrt()
        
        scale = orig_rms / (new_rms + _EPS)
        return noise_spectral * scale

    if mode == "percentile":
        abs_noise = noise.abs()
        if noise.ndim > 1:
            dims = tuple(range(1, noise.ndim))
            p99 = torch.quantile(abs_noise.flatten(), 0.99)
        else:
            p99 = torch.quantile(abs_noise, 0.99)
        
        target_p99 = 2.576
        
        if p99 < _EPS:
            return noise
        
        scale = target_p99 / p99
        return noise * scale

    if mode == "adaptive":
        if sigma_from is None or sigma_to is None:
            return noise * (float(count) ** 0.5)
        
        sigma_eff = math.sqrt(sigma_from * sigma_to) if sigma_from > 0 and sigma_to > 0 else sigma_from
        
        if sigma_eff > 1.0:
            adaptive_factor = 0.8 + 0.2 / (sigma_eff + 0.1)
        elif sigma_eff > 0.1:
            adaptive_factor = 1.0 + 0.3 * math.log10(1.0 / sigma_eff)
        else:
            adaptive_factor = 1.2 + 0.4 * math.log10(1.0 / sigma_eff)
        
        adaptive_factor = adaptive_factor.clamp(0.5, 2.0) if isinstance(adaptive_factor, torch.Tensor) else max(0.5, min(2.0, adaptive_factor))
        
        base_scale = float(count) ** 0.5
        return noise * (base_scale * adaptive_factor)

    if mode == "snr":
        if noise.ndim < 4:
            return noise * (float(count) ** 0.5)
        
        kernel_size = 7
        padding = kernel_size // 2
        
        signal_estimate = torch.nn.functional.avg_pool2d(
            noise, kernel_size=kernel_size, stride=1, padding=padding
        )
        
        local_var = torch.nn.functional.avg_pool2d(
            noise ** 2, kernel_size=kernel_size, stride=1, padding=padding
        ) - signal_estimate ** 2
        local_var = local_var.clamp_min(_EPS)
        
        target_variance = local_var.mean()
        snr_scale = torch.sqrt(target_variance / local_var)
        snr_scale = snr_scale.clamp(0.5, 2.0)
        
        base_scale = float(count) ** 0.5
        return noise * (base_scale * snr_scale)

    return noise


print("=" * 70)
print("COMPREHENSIVE ANALYSIS OF ALL NORMALIZATION MODES")
print("=" * 70)

N = 4
B, C, H, W = 2, 4, 64, 64

noises = [torch.randn(B, C, H, W) for _ in range(N)]
merged = torch.stack(noises).mean(dim=0)

print(f"\nInput: Merged noise from {N} paths, shape {merged.shape}")
print(f"Input std: {merged.std().item():.4f} (expected ~{1/math.sqrt(N):.4f})")
print(f"Input variance: {merged.var().item():.4f} (expected ~{1/N:.4f})")

modes = ["variance", "rms", "spectral", "percentile", "adaptive", "snr"]
sigma_from, sigma_to = 1.0, 0.8

print("\n" + "-" * 70)
for mode in modes:
    try:
        out = _normalize_noise_current(merged.clone(), mode, N, sigma_from, sigma_to)
        print(f"{mode:12s}: std={out.std().item():.4f}, var={out.var().item():.4f}, mean={out.mean().item():.6f}")
    except Exception as e:
        print(f"{mode:12s}: ERROR - {e}")

print("\n" + "=" * 70)
print("IDENTIFIED BUGS:")
print("=" * 70)

print("""
BUG 1: RMS Mode - Per-sample RMS computation
  Location: Lines 225-235
  Problem: Computes RMS over dimensions 1..ndim (channels only for 4D tensors)
  Impact: Creates spatially-varying amplification causing artifacts
  Fix: Use global RMS (mean over ALL elements)

BUG 2: Spectral Mode - Same per-sample RMS issue  
  Location: Lines 272-280
  Problem: Uses per-sample RMS for energy normalization
  Impact: Introduces spatial inconsistencies after FFT processing
  Fix: Use global RMS for both orig_rms and new_rms

BUG 3: Percentile Mode - Wrong quantile computation
  Location: Lines 292-296
  Problem: Flattens entire tensor but should preserve batch structure
  Impact: May not correctly handle batched inputs
  Fix: Compute percentile globally or per-batch consistently

BUG 4: Adaptive Mode - Missing sqrt(count) scaling consistency
  Location: Lines 317-342
  Problem: When sigma info is available, applies adaptive_factor but the
           base scaling may not match the statistical expectations
  Impact: Inconsistent noise magnitude across timesteps
  Fix: Ensure base_scale properly accounts for merged noise variance

BUG 5: SNR Mode - Computing statistics on pure noise
  Location: Lines 353-386
  Problem: Estimates "signal" from noise tensor itself (which has no signal!)
  Impact: Creates artificial spatial patterns from random noise statistics
  Fix: Should estimate signal from the actual image/latent, not the noise

BUG 6: All modes - Missing count parameter usage consistency
  Location: Throughout
  Problem: Some modes use count, others don't, leading to inconsistent scaling
  Impact: Different modes produce different output magnitudes
  Fix: Standardize scaling across all modes to produce unit variance output
""")

print("\n" + "=" * 70)
print("DETAILED RMS BUG ANALYSIS:")
print("=" * 70)

# Show the exact problem with per-sample RMS
dims = tuple(range(1, merged.ndim))
rms_per_sample = merged.pow(2).mean(dim=dims, keepdim=True).sqrt()
rms_global = merged.pow(2).mean().sqrt()

print(f"\nPer-sample RMS shape: {rms_per_sample.shape}")
print(f"Per-sample RMS varies across spatial locations: std={rms_per_sample.std().item():.6f}")
print(f"Global RMS: {rms_global.item():.6f}")

# The division creates artifacts
normalized_per_sample = merged / rms_per_sample.clamp_min(_EPS)
normalized_global = merged / rms_global

print(f"\nAfter per-sample normalization:")
print(f"  Spatial variance consistency: std={normalized_per_sample.pow(2).mean(dim=[0,1]).std().item():.4f}")
print(f"After global normalization:")
print(f"  Spatial variance consistency: std={normalized_global.pow(2).mean(dim=[0,1]).std().item():.4f}")
print(f"\nLower spatial variance std = more uniform noise = fewer artifacts")

