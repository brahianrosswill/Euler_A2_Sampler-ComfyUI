"""Understanding the math behind noise normalization."""
import torch
import math

N = 4
B, C, H, W = 2, 4, 64, 64

# Generate N independent noise tensors (each with unit variance)
noises = [torch.randn(B, C, H, W) for _ in range(N)]

print("=" * 70)
print("UNDERSTANDING NOISE NORMALIZATION MATHEMATICS")
print("=" * 70)

for i, n in enumerate(noises):
    print(f"Noise {i}: std={n.std().item():.4f}, var={n.var().item():.4f}")

# Merge by averaging
merged = torch.stack(noises).mean(dim=0)

print(f"\nMerged (mean of {N} noises):")
print(f"  std={merged.std().item():.4f} (expected: {1/math.sqrt(N):.4f})")
print(f"  var={merged.var().item():.4f} (expected: {1/N:.4f})")
print(f"  RMS={merged.pow(2).mean().sqrt().item():.4f} (expected: ~{1/math.sqrt(N):.4f})")

print("\n" + "-" * 70)
print("VARIANCE MODE:")
print("-" * 70)
var_out = merged * math.sqrt(N)
print(f"After multiplying by sqrt({N}) = {math.sqrt(N):.4f}:")
print(f"  std={var_out.std().item():.4f} (target: 1.0)")
print(f"  var={var_out.var().item():.4f} (target: 1.0)")

print("\n" + "-" * 70)
print("RMS MODE - CORRECT APPROACH:")
print("-" * 70)
rms_merged = merged.pow(2).mean().sqrt()
print(f"Merged RMS: {rms_merged.item():.4f}")
print(f"Expected RMS for merged noise: ~{1/math.sqrt(N):.4f}")

rms_out = merged / rms_merged
print(f"\nAfter dividing by RMS:")
print(f"  New RMS: {rms_out.pow(2).mean().sqrt().item():.4f} (should be 1.0)")
print(f"  std: {rms_out.std().item():.4f} (should be ~1.0)")
print(f"  var: {rms_out.var().item():.4f} (should be ~1.0)")

print("\nKEY INSIGHT:")
print("For zero-mean Gaussian noise:")
print("  E[X^2] = Var(X) + E[X]^2 = Var(X) + 0 = Var(X)")
print("  RMS = sqrt(E[X^2]) = sqrt(Var(X)) = std")
print("\nSo for merged noise with var=1/N:")
print("  RMS ≈ sqrt(1/N) = 1/sqrt(N)")
print("  Dividing by RMS gives: new_std = (1/sqrt(N)) / (1/sqrt(N)) = 1")
print("  This ALREADY produces unit variance - no extra scaling needed!")

print("\n" + "=" * 70)
print("CONCLUSION:")
print("=" * 70)
print("""
Variance mode: multiply by sqrt(N) → variance goes from 1/N to 1 ✓
RMS mode: divide by RMS → variance goes from 1/N to 1 ✓

Both should produce std≈1.0, var≈1.0!

The test was failing because it expected BOTH modes to have sqrt(N) scaling,
but RMS normalization already achieves the correct scaling inherently.
""")

