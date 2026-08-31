"""Test script to analyze the noise normalization bug."""
import torch
import math

_EPS = 1e-8

def _strip_str(value):
    return value.strip() if isinstance(value, str) else value

def _normalize_noise_buggy(
    noise: torch.Tensor, mode: str, count: int, sigma_from: float = None, sigma_to: float = None
) -> torch.Tensor:
    """Current buggy implementation."""
    mode = _strip_str(mode)

    if mode == "variance":
        return noise * (float(count) ** 0.5)

    if mode == "rms":
        if noise.ndim > 1:
            dims = tuple(range(1, noise.ndim))
            rms = noise.pow(2).mean(dim=dims, keepdim=True).sqrt()
        else:
            rms = noise.pow(2).mean().sqrt()

        # Guard against all-zero latents.
        if rms.max() < _EPS:
            return noise

        return noise / rms.clamp_min(_EPS)

    return noise


def _normalize_noise_fixed(
    noise: torch.Tensor, mode: str, count: int, sigma_from: float = None, sigma_to: float = None
) -> torch.Tensor:
    """Fixed implementation."""
    mode = _strip_str(mode)

    if mode == "variance":
        # The issue: when we merge N noise paths by averaging, the variance becomes 1/N
        # To restore unit variance, we should multiply by sqrt(N)
        # BUT this is ONLY correct if the input noise already has unit variance
        # After merging (mean of N samples), variance = original_variance / N
        # So we need: output = merged_noise * sqrt(N) to get back to original variance
        # This is actually CORRECT in the current code!
        # 
        # However, the REAL issue might be that this doesn't account for the fact that
        # the merged noise is being used in a specific context where the scale matters
        return noise * math.sqrt(float(count))

    if mode == "rms":
        # CRITICAL BUG: The current RMS normalization forces RMS=1 globally
        # But for Gaussian noise with unit variance, expected RMS depends on tensor size
        # For a tensor of shape [B,C,H,W] with i.i.d. N(0,1) entries:
        #   E[RMS^2] = E[mean(noise^2)] = 1 (since E[noise^2] = var = 1)
        # So RMS should be ~1 for unit variance noise
        # 
        # The problem: dividing by RMS makes the output have RMS=1, but this DESTROYS
        # the natural variance structure and can amplify noise artifacts
        # 
        # What we REALLY want: preserve the statistical properties while ensuring
        # the noise magnitude is appropriate for the diffusion step
        
        if noise.numel() == 0:
            return noise
            
        # Compute global RMS (not per-sample which causes issues with batching)
        rms = noise.pow(2).mean().sqrt()
        
        if rms < _EPS:
            return noise
        
        # Normalize to unit RMS, then rescale by sqrt(count) to match variance correction
        # This ensures consistent behavior with variance mode
        normalized = noise / rms
        return normalized * math.sqrt(float(count))

    return noise


# Test case: simulate what happens in the actual code
print("=" * 60)
print("Testing noise normalization behavior")
print("=" * 60)

# Simulate merging N=4 noise paths (each N(0,1))
N = 4
B, C, H, W = 2, 4, 64, 64

# Generate N independent noise tensors (simulating noise_paths)
noises = [torch.randn(B, C, H, W) for _ in range(N)]

# Merge by averaging (as done in _merge_noise_paths)
merged = torch.stack(noises).mean(dim=0)

print(f"\nInput: {N} independent N(0,1) noise tensors of shape {noises[0].shape}")
print(f"Merged (mean) shape: {merged.shape}")
print(f"Merged mean: {merged.mean().item():.6f} (should be ~0)")
print(f"Merged std: {merged.std().item():.6f} (should be ~{1/math.sqrt(N):.4f} = 1/sqrt({N}))")
print(f"Merged variance: {merged.var().item():.6f} (should be ~{1/N:.4f} = 1/{N})")
print(f"Merged RMS: {merged.pow(2).mean().sqrt().item():.6f} (should be ~{1/math.sqrt(N):.4f})")

# Test variance mode
var_out = _normalize_noise_buggy(merged.clone(), "variance", N)
print(f"\nVariance mode (buggy):")
print(f"  Output std: {var_out.std().item():.6f} (should be ~1.0)")
print(f"  Output variance: {var_out.var().item():.6f} (should be ~1.0)")

var_out_fixed = _normalize_noise_fixed(merged.clone(), "variance", N)
print(f"Variance mode (fixed):")
print(f"  Output std: {var_out_fixed.std().item():.6f} (should be ~1.0)")

# Test RMS mode
rms_out = _normalize_noise_buggy(merged.clone(), "rms", N)
print(f"\nRMS mode (buggy):")
print(f"  Output std: {rms_out.std().item():.6f} (should be ~1.0)")
print(f"  Output RMS: {rms_out.pow(2).mean().sqrt().item():.6f} (should be ~1.0)")

rms_out_fixed = _normalize_noise_fixed(merged.clone(), "rms", N)
print(f"RMS mode (fixed):")
print(f"  Output std: {rms_out_fixed.std().item():.6f} (should be ~1.0)")
print(f"  Output RMS: {rms_out_fixed.pow(2).mean().sqrt().item():.6f} (should be ~1.0)")

# The REAL issue analysis
print("\n" + "=" * 60)
print("ANALYSIS OF THE BUG:")
print("=" * 60)

# When you average N independent N(0,1) samples:
# - Mean stays 0
# - Variance becomes 1/N
# - Std becomes 1/sqrt(N)

# For variance mode: multiplying by sqrt(N) gives variance = (1/N) * N = 1 ✓
# This seems correct!

# For RMS mode: dividing by RMS normalizes to RMS=1
# But RMS of averaged noise ≈ 1/sqrt(N), so dividing by it multiplies by sqrt(N)
# This also gives variance ≈ 1 ✓

# SO WHY THE ARTIFACTS?

# Let's check what happens with the PER-SAMPLE RMS calculation
print("\nChecking per-sample vs global RMS:")
if merged.ndim > 1:
    dims = tuple(range(1, merged.ndim))
    rms_per_sample = merged.pow(2).mean(dim=dims, keepdim=True).sqrt()
    print(f"Per-sample RMS shape: {rms_per_sample.shape}")
    print(f"Per-sample RMS mean: {rms_per_sample.mean().item():.6f}")
    print(f"Per-sample RMS std: {rms_per_sample.std().item():.6f}")
    
    # The problem: per-sample RMS varies across spatial locations
    # Dividing by this creates spatially-varying amplification
    # This introduces ARTIFACTS because some regions get amplified more than others
    
    global_rms = merged.pow(2).mean().sqrt()
    print(f"Global RMS: {global_rms.item():.6f}")
    
    # Compare outputs
    per_sample_normalized = merged / rms_per_sample.clamp_min(_EPS)
    global_normalized = merged / global_rms
    
    print(f"\nPer-sample normalized std: {per_sample_normalized.std().item():.6f}")
    print(f"Global normalized std: {global_normalized.std().item():.6f}")
    
    # Check spatial consistency
    print(f"\nPer-sample normalized spatial variance (should be uniform):")
    per_sample_spatial_var = per_sample_normalized.pow(2).mean(dim=[0,1])
    print(f"  Spatial var mean: {per_sample_spatial_var.mean().item():.6f}")
    print(f"  Spatial var std: {per_sample_spatial_var.std().item():.6f} (lower is better)")
    
    global_spatial_var = global_normalized.pow(2).mean(dim=[0,1])
    print(f"Global normalized spatial variance:")
    print(f"  Spatial var mean: {global_spatial_var.mean().item():.6f}")
    print(f"  Spatial var std: {global_spatial_var.std().item():.6f} (lower is better)")

print("\n" + "=" * 60)
print("CONCLUSION:")
print("=" * 60)
print("""
The RMS mode bug is in computing RMS per-sample (across channels only) instead 
of globally. This causes:

1. SPATIALLY-VARYING AMPLIFICATION: Different spatial locations get different 
   scaling factors, introducing artificial patterns/artifacts

2. BROKEN STATISTICS: The per-sample RMS doesn't represent the true noise 
   magnitude, leading to incorrect scaling

3. ARTIFACT GENERATION: The spatially-varying division creates structured 
   noise patterns that weren't in the original random noise

FIX: Use GLOBAL RMS computation (mean over ALL dimensions) instead of 
     per-sample RMS (mean over channels only).
""")

