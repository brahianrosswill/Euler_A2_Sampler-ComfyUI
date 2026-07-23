# Sampler Accuracy Improvements - Implementation Summary

This document summarizes the accuracy improvements added to the Euler-A2 sampler.

## New Features Added

### 1. Milstein Integration Method
**Purpose**: Improve SDE integration accuracy by accounting for drift curvature.

- **Implementation**: Added `milstein` method to the `_ode_step()` function
- **How it works**: Uses a second-order Taylor expansion that includes a correction term based on the derivative of the drift field
- **Benefits**: Better accuracy for larger step sizes, especially in regions with high curvature
- **Usage**: Set `method="milstein"` in the sampler options
- **Cost**: 2 model evaluations per step (same as other 2nd-order methods)

### 2. Langevin Corrector
**Purpose**: Refine samples after each predictor step using score-based gradient descent.

- **Modes**:
  - `none`: No corrector (default, backward compatible)
  - `langevin`: Fixed step size Langevin dynamics
  - `langevin_dynamic`: Adaptive step size based on local gradient magnitude

- **Parameters**:
  - `corrector_steps`: Number of corrector iterations per step (0-8)
  - `corrector_eta`: Step size multiplier (0.0-2.0, default 0.5)

- **How it works**: After each sampling step, applies Langevin MCMC refinement:
  ```
  x += step_size * score + noise * sqrt(2 * step_size)
  ```
  where score ≈ (denoised - x) / σ²

- **Benefits**: Reduces discretization error, improves sample quality near convergence
- **Cost**: Additional model evaluations (corrector_steps per sampling step)

### 3. Antithetic Sampling (Variance Reduction)
**Purpose**: Reduce Monte Carlo variance through symmetric noise pairing.

- **Implementation**: `_antithetic_noise()` helper function
- **How it works**: For each random noise draw ε, also uses -ε, ensuring the combined noise distribution is perfectly symmetric around zero
- **Benefits**: 
  - Reduced variance in noise estimation
  - More stable sampling trajectories
  - Better convergence properties
- **Usage**: Set `variance_reduction="antithetic"`
- **Note**: Generates noise in pairs, so effective path count is even

## Updated Configuration Options

### New Constants
```python
METHODS = ("euler", "midpoint", "ralston", "heun", "dpm2", "rk3", "rk4", "ab2", "er_sde", "milstein")
CORRECTOR_MODES = ("none", "langevin", "langevin_dynamic")
VARIANCE_REDUCTION_MODES = ("none", "antithetic")
```

### New Node Parameters
| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `corrector` | string | "none" | - | Post-step corrector mode |
| `corrector_steps` | int | 1 | 0-8 | Langevin iterations per step |
| `corrector_eta` | float | 0.5 | 0.0-2.0 | Langevin step size multiplier |
| `variance_reduction` | string | "none" | - | Variance reduction technique |

## Recommended Usage Patterns

### High-Quality Generation (Accuracy Priority)
```
method = "milstein" or "rk4"
substeps = 2-4
corrector = "langevin_dynamic"
corrector_steps = 1-2
corrector_eta = 0.3-0.5
noise_paths = 3-4
merge_mode = "mean"
noise_normalization = "rms"
```

### Fast Generation (Speed Priority)
```
method = "euler" or "heun"
substeps = 1
corrector = "none"
variance_reduction = "antithetic"
noise_paths = 2
```

### Balanced Quality/Speed
```
method = "er_sde" or "dpm2"
substeps = 1-2
corrector = "langevin"
corrector_steps = 1
corrector_eta = 0.5
noise_paths = 2
variance_reduction = "antithetic"
```

## Theoretical Background

### Milstein Scheme
The Milstein method is a strong order 1.0 numerical scheme for SDEs that extends the Euler-Maruyama method by including a correction term:

```
X_{n+1} = X_n + a(X_n)h + b(X_n)ΔW + 0.5*b(X_n)*b'(X_n)*((ΔW)² - h)
```

For the probability-flow ODE, this translates to capturing the curvature of the drift field, improving accuracy especially when the denoising function has significant nonlinearity.

### Langevin Dynamics
Langevin MCMC uses the score function (gradient of log-density) to guide samples toward high-probability regions:

```
dx = ∇log p(x) dt + √2 dW
```

In diffusion models, the score is approximated as:
```
∇log p_σ(x) ≈ (D(x) - x) / σ²
```

where D(x) is the denoiser output.

### Antithetic Variates
By pairing each random sample ε with its negation -ε, the variance of estimators is reduced:

```
Var[(f(ε) + f(-ε))/2] ≤ Var[f(ε)]
```

This is particularly effective when f has monotonic or symmetric components.

## Backward Compatibility

All new parameters have default values that preserve the original Euler-A2 behavior:
- `corrector="none"` - no corrector applied
- `variance_reduction="none"` - standard noise sampling
- `method="euler"` (if explicitly set) - original integration method

Existing workflows will continue to work without modification.

## Performance Considerations

| Feature | Additional Model Evaluations | Memory Impact |
|---------|------------------------------|---------------|
| Milstein | +1 per step (2 total) | None |
| Langevin corrector | +corrector_steps per step | None |
| Antithetic sampling | None (same paths) | Minimal (temporary pair storage) |

## Future Improvement Opportunities

Based on research, additional enhancements could include:

1. **Adaptive step sizing** - Dynamic sigma adjustment based on local error estimates
2. **DPM-Solver++** - Specialized high-order solver for diffusion ODEs
3. **EDM sigma parameters** - Churn, tmin, tmax controls from the EDM paper
4. **Stochastic Runge-Kutta** - Higher-order strong SDE integrators
5. **Richardson extrapolation** - Order elevation through multi-resolution combination

## References

- Milstein, G.N. (1974). "Approximate Integration of Stochastic Differential Equations"
- Song et al. (2020). "Score-Based Generative Modeling through Stochastic Differential Equations"
- Karras et al. (2022). "Elucidating the Design Space of Diffusion-Based Generative Models (EDM)"
- Lu et al. (2022). "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling"
