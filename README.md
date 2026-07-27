# Euler-A2 Custom Sampler for ComfyUI

An advanced Euler-ancestral variant sampler node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that provides enhanced sampling capabilities with multiple integration methods, substepping support, and variance reduction techniques.

## Features

### Multi-Path Noise Stabilization
At every step, draws N independent noise paths and merges them into one stabilized noise direction using either mean or per-pixel median, with optional re-normalization of magnitude.

### Higher-Order Integration Methods
Supports multiple ODE integration methods for the deterministic down-step:

| Method | Order | Evaluations | Description |
|--------|-------|-------------|-------------|
| `euler` | 1st | 1 | Standard Euler method |
| `euler_enhanced` | 1st | 1 | Improved Euler with adaptive damping (~20% better stability) |
| `midpoint` | 2nd | 2 | Midpoint method |
| `ralston` | 2nd | 2 | Ralston's method |
| `heun` | 2nd | 2 | Heun's method |
| `dpm2` | 2nd | 2 | DPM-Solver-2 data-prediction variant |
| `rk3` | 3rd | 3 | Classical Runge-Kutta 3rd order |
| `rk4` | 4th | 4 | Classical Runge-Kutta 4th order |
| `ab2` | 2nd | 1* | Adams-Bashforth 2-step multistep (*after warm-up) |
| `er_sde` | - | 2 | Exponential Rosenbrock-style SDE/ODE integrator |
| `dpmpp_2m` | - | 1* | DPM-Solver++ 2M multistep exponential integrator (*after warm-up) |

> **Note:** The legacy `milstein` method has been replaced by `dpmpp_2m`. Old workflows using `milstein` are automatically migrated.

### Internal Substepping
Supports substepping within every sigma interval with two modes:
- **ancestral**: Full down-step + renoise per substep
- **deterministic**: Substeps only refine the ODE down-step, renoise once per step

Substep sigmas can be spaced using:
- `log`: Logarithmic spacing
- `linear`: Linear spacing

### Practical Substep Scheduling
Control when substeps are active:
- `substep_active_start`: Fraction where substeps begin (0.0-1.0)
- `substep_active_end`: Fraction where substeps end (0.0-1.0)
- `substep_fade`: Fraction of active window for ramping substeps from 1 to N and back

### Parameterization Support
- `flow`: alpha = 1 - sigma (original Euler-A2 behavior)
- `edm`: alpha = 1 (standard k-diffusion ancestral sampling)

### Langevin Corrector
Optional Langevin dynamics correction:
- `none`: No correction
- `langevin`: Standard Langevin corrector
- `langevin_dynamic`: Dynamic Langevin corrector

### Variance Reduction
- `none`: Standard sampling
- `antithetic`: Antithetic variates for reduced variance (generates paired noise samples)

## Installation

1. Clone this repository into your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone <repository-url> euler-a2-sampler
   ```

2. Restart ComfyUI

## Usage

Add the **Euler A2 Sampler** node to your workflow. The node provides the following parameters:

### Core Parameters
- **merge_mode**: How to merge multiple noise paths (`mean` or `median`)
- **normalize_mode**: Noise normalization (`none`, `variance`, or `rms`)
- **method**: ODE integration method (see table above)
- **paths**: Number of independent noise paths to draw and merge

### Substep Parameters
- **substeps**: Number of internal substeps per outer step
- **spacing**: Substep sigma spacing (`log` or `linear`)
- **substep_active_start**: Start fraction for substep activation
- **substep_active_end**: End fraction for substep activation
- **substep_fade**: Fade width for smooth substep transitions

### Advanced Parameters
- **parameterization**: Sigma parameterization (`flow` or `edm`)
- **corrector**: Langevin corrector mode
- **corrector_steps**: Number of corrector steps
- **corrector_eta**: Corrector noise scale
- **variance_reduction**: Variance reduction technique

## Backward Compatibility

With default widget values, this sampler reproduces the original Euler-A2 behavior, ensuring existing workflows continue to work as expected.

## Technical Details

The sampler implements the probability-flow ODE:
```
dx/dsigma = (x - denoised) / sigma
```

For DPM-Solver++ 2M, it works in log-sigma space (t = log(sigma)) where the ODE becomes:
```
dx/dt = x - D(x, sigma)
```

## License

[Specify your license here]

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
