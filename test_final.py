"""Final test of the fixed normalization implementation."""
import torch
import math

_EPS = 1e-8

def _strip_str(value):
    return value.strip() if isinstance(value, str) else value

def _normalize_noise(
    noise: torch.Tensor, mode: str, count: int, sigma_from: float = None, sigma_to: float = None
) -> torch.Tensor:
    """Fixed implementation matching nodes.py."""
    mode = _strip_str(mode)

    if mode == "variance":
        return noise * math.sqrt(float(count))

    if mode == "rms":
        # Use GLOBAL RMS (not per-sample) to avoid spatial artifacts
        if noise.numel() == 0:
            return noise
        
        rms = noise.pow(2).mean().sqrt()
        
        if rms < _EPS:
            return noise
        
        # Dividing by RMS inherently normalizes to unit variance
        return noise / rms

    if mode == "spectral":
        noise_fft = torch.fft.fftn(noise.float())
        
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
        
        orig_rms = noise.pow(2).mean().sqrt()
        new_rms = noise_spectral.pow(2).mean().sqrt()
        
        scale = orig_rms / (new_rms + _EPS)
        return noise_spectral * scale * math.sqrt(float(count))

    if mode == "percentile":
        abs_noise = noise.abs()
        p99 = torch.quantile(abs_noise.flatten(), 0.99)
        
        target_p99 = 2.576
        
        if p99 < _EPS:
            return noise
        
        scale = target_p99 / p99
        return noise * scale

    if mode == "adaptive":
        if sigma_from is None or sigma_to is None:
            return noise * math.sqrt(float(count))
        
        sigma_eff = math.sqrt(sigma_from * sigma_to) if sigma_from > 0 and sigma_to > 0 else sigma_from
        
        if sigma_eff > 1.0:
            adaptive_factor = 0.8 + 0.2 / (sigma_eff + 0.1)
        elif sigma_eff > 0.1:
            adaptive_factor = 1.0 + 0.3 * math.log10(1.0 / sigma_eff)
        else:
            adaptive_factor = 1.2 + 0.4 * math.log10(1.0 / sigma_eff)
        
        if isinstance(adaptive_factor, torch.Tensor):
            adaptive_factor = adaptive_factor.clamp(0.5, 2.0)
        else:
            adaptive_factor = max(0.5, min(2.0, adaptive_factor))
        
        base_scale = math.sqrt(float(count))
        return noise * (base_scale * adaptive_factor)

    if mode == "snr":
        if noise.ndim < 4:
            return noise * math.sqrt(float(count))
        
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
        
        base_scale = math.sqrt(float(count))
        return noise * (base_scale * snr_scale)

    return noise


print("=" * 70)
print("FINAL TEST OF FIXED NORMALIZATION")
print("=" * 70)

N = 4
B, C, H, W = 2, 4, 64, 64

noises = [torch.randn(B, C, H, W) for _ in range(N)]
merged = torch.stack(noises).mean(dim=0)

print(f"\nInput: Merged noise from {N} paths")
print(f"Input std: {merged.std().item():.4f} (expected ~{1/math.sqrt(N):.4f})")

modes = ["none", "variance", "rms", "spectral", "percentile", "adaptive", "snr"]
sigma_from, sigma_to = 1.0, 0.8

print("\n" + "-" * 70)
for mode in modes:
    out = _normalize_noise(merged.clone(), mode, N, sigma_from, sigma_to)
    print(f"{mode:12s}: std={out.std().item():.4f}, var={out.var().item():.4f}")

print("\n" + "=" * 70)
print("EXPECTED RESULTS:")
print("=" * 70)
print("""
- none:       std≈0.5 (keeps merged noise as-is)
- variance:   std≈1.0 (multiplies by sqrt(N)=2)
- rms:        std≈1.0 (divides by RMS, which is ≈0.5)
- spectral:   std≈1.0 (FFT shaping + sqrt(N) scaling)
- percentile: std≈1.0 (scales to match 99th percentile of N(0,1))
- adaptive:   std≈1.0 (sqrt(N) × adaptive factor near 1.0)
- snr:        std≈1.0 (sqrt(N) × SNR-based spatial scaling)

All modes except 'none' should produce std close to 1.0!
""")

print("=" * 70)
print("CRITICAL FIX SUMMARY:")
print("=" * 70)
print("""
1. RMS MODE: Changed from per-sample RMS to GLOBAL RMS
   - OLD: rms = noise.pow(2).mean(dim=dims, keepdim=True).sqrt()  # dims=(1,2,3)
   - NEW: rms = noise.pow(2).mean().sqrt()  # Global mean
   
   This eliminates spatially-varying amplification that caused artifacts.

2. SPECTRAL MODE: Changed to use GLOBAL RMS for energy normalization
   - Same fix as RMS mode to ensure consistent scaling

3. PERCENTILE MODE: Simplified to not double-count sqrt(N) scaling
   - Percentile targeting already produces correct variance

4. ALL MODES: Consistent use of math.sqrt(float(count)) instead of count**0.5
""")

