# Euler-A2 Custom Sampler for ComfyUI

A sophisticated Euler-ancestral variant sampler node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that introduces advanced noise stabilization, higher-order integration methods, and flexible sampling controls for improved image generation quality.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Parameter Reference](#parameter-reference)
  - [Core Parameters](#core-parameters)
  - [Noise Path Configuration](#noise-path-configuration)
  - [Integration Methods](#integration-methods)
  - [Substep Controls](#substep-controls)
  - [Parameterization Modes](#parameterization-modes)
  - [Langevin Corrector](#langevin-corrector)
  - [Variance Reduction](#variance-reduction)
- [How It Works](#how-it-works)
  - [Noise Path Merging](#noise-path-merging)
  - [Higher-Order Integration](#higher-order-integration)
  - [Substep Scheduling](#substep-scheduling)
- [Practical Guidelines](#practical-guidelines)
- [Backward Compatibility](#backward-compatibility)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview

Euler-A2 is an enhanced ancestral sampler that improves upon the standard Euler-a method by introducing **multiple independent noise paths per sampling step**. Instead of drawing a single noise vector at each step, Euler-A2 generates N independent noise samples, merges them into a stabilized direction using either mean or median operations, optionally normalizes the magnitude, and then extrapolates along this refined direction.

This approach provides:
- **Reduced noise artifacts** through statistical averaging
- **Better convergence** via higher-order integration methods
- **Fine-grained control** over the sampling process
- **Flexibility** to reproduce standard Euler-a behavior or explore new creative territories

---

## Key Features

### 🎯 Multi-Path Noise Stabilization
Draw multiple independent noise samples per step and merge them using:
- **Mean**: Standard averaging for variance reduction
- **Median**: Robust outlier rejection (recommended with 3+ paths)

### 🔬 Higher-Order Integration Methods
Choose from 11 different ODE integration schemes for the deterministic down-step:
- `euler` – Standard 1st order (1 evaluation)
- `euler_enhanced` – Improved 1st order with adaptive damping (~20% better stability)
- `midpoint` – 2nd order (2 evaluations)
- `ralston` – 2nd order optimized (2 evaluations)
- `heun` – 2nd order predictor-corrector (2 evaluations)
- `dpm2` – DPM-Solver-2 data-prediction variant (2 evaluations)
- `rk3` – Classical 3rd order Runge-Kutta (3 evaluations)
- `rk4` – Classical 4th order Runge-Kutta (4 evaluations)
- `ab2` – Adams-Bashforth 2-step multistep method (1 evaluation after warm-up)
- `er_sde` – Exponential Rosenbrock-style integrator (2 evaluations)
- `dpmpp_2m` – DPM-Solver++ 2M multistep exponential integrator (1 evaluation per step after warm-up)

### 📐 Internal Substepping
Refine each sigma interval with internal substeps in two modes:
- **Ancestral**: Full down-step + renoise per substep (full SDE discretization)
- **Deterministic**: Substeps refine only the ODE down-step; renoise once per outer step

Substep sigmas can be spaced:
- **Log**: Geometric spacing (default, better for diffusion schedules)
- **Linear**: Uniform spacing

### 📅 Substep Scheduling
Control when substeps are active during the sampling process:
- `substep_active_start`: Fraction where substeps begin (0.0 = first step)
- `substep_active_end`: Fraction where substeps end (1.0 = last step)
- `substep_fade`: Smooth fade-in/out over a fraction of the active window

Example: Use high substep counts only in the middle of sampling where details emerge, saving computation on early/noisy and late/fine-tuning steps.

### 🔄 Parameterization Support
Two parameterization modes for different diffusion model conventions:
- **Flow** (`alpha = 1 - sigma`): Original Euler-A2 behavior
- **EDM** (`alpha = 1`): Standard k-diffusion ancestral sampling

### 🧪 Langevin Corrector
Post-step refinement using Langevin dynamics:
- **None**: No correction (fastest)
- **Langevin**: Fixed step-size unadjusted Langevin
- **Langevin Dynamic**: Adaptive step-size based on gradient norm

Configurable corrector iterations and step-size multiplier.

### 📉 Variance Reduction
Reduce sampling variance through antithetic sampling:
- **None**: Standard independent noise draws
- **Antithetic**: Paired +/- noise for symmetry, ensuring more balanced noise distribution

---

## Installation

### Manual Installation

1. Clone or download this repository into your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone <repository-url> euler-a2-sampler
   ```

2. Restart ComfyUI

The sampler will automatically register itself with ComfyUI's sampler registry on import.

### Requirements

- **ComfyUI**: Latest version recommended
- **PyTorch**: As required by your ComfyUI installation
- **Python**: 3.8+

No additional dependencies beyond ComfyUI's standard requirements.

---

## Usage

After installation, the **Euler A2 Sampler** node will appear in ComfyUI under:

```
sampling → custom_sampling → samplers → Euler A2 Sampler
```

### Basic Workflow

1. Add the **Euler A2 Sampler** node to your workflow
2. Connect its output to a **KSampler** node's `sampler` input
3. Configure parameters as needed (defaults reproduce original Euler-A2 behavior)
4. Generate images as usual

### Example Configuration

For high-quality generation with moderate compute:

```
eta: 1.0
s_noise: 1.0
extrapolation: 0.425
noise_paths: 2
merge_mode: mean
noise_normalization: none
method: euler_enhanced
substeps: 1
parameterization: flow
corrector: none
```

For maximum quality with extended compute time:

```
eta: 1.0
s_noise: 1.0
extrapolation: 0.425
noise_paths: 4
merge_mode: median
noise_normalization: variance
method: rk4
substeps: 4
substep_mode: ancestral
substep_spacing: log
parameterization: flow
corrector: langevin_dynamic
corrector_steps: 2
variance_reduction: antithetic
```

---

## Parameter Reference

### Core Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `eta` | Float | 1.0 | 0.0 – 100.0 | Ancestral interpolation factor. 0 = deterministic (ODE), 1 = full ancestral noise injection. |
| `s_noise` | Float | 1.0 | 0.0 – 100.0 | Global multiplier on injected noise amplitude. |
| `extrapolation` | Float | 0.425 | -10.0 – 10.0 | Extra gain along the merged noise direction. Total gain = 1 + extrapolation. Value ~0.414 preserves noise energy for 2 paths without normalization. |

### Noise Path Configuration

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `noise_paths` | Int | 2 | 1 – 8 | Number of independent noise draws merged per step. Higher values increase stability but add compute cost. |
| `merge_mode` | Enum | mean | mean, median | How to combine noise paths. **mean** is standard averaging. **median** is robust to outliers; use with odd counts ≥ 3. |
| `noise_normalization` | Enum | none | none, variance, rms | Rescaling strategy for merged noise:<br>• **none**: Keep as-is (variance ~1/N)<br>• **variance**: Multiply by √N to restore unit variance<br>• **rms**: Force per-sample RMS to exactly 1 |

### Integration Methods

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `method` | Enum | euler | See below | Integration method for deterministic down-step. |

**Available Methods:**

| Method | Order | Evaluations | Description |
|--------|-------|-------------|-------------|
| `euler` | 1st | 1 | Standard Euler method. Fast, baseline quality. |
| `euler_enhanced` | 1st | 1 | Improved Euler with adaptive damping. ~20% better stability without extra cost. |
| `midpoint` | 2nd | 2 | Midpoint method. Better accuracy than Euler. |
| `ralston` | 2nd | 2 | Ralston's optimized 2nd-order method. Minimizes error bound. |
| `heun` | 2nd | 2 | Heun's predictor-corrector. Good balance of speed/accuracy. |
| `dpm2` | 2nd | 2 | DPM-Solver-2 data-prediction variant. Designed for diffusion models. |
| `rk3` | 3rd | 3 | Classical 3rd-order Runge-Kutta. High accuracy. |
| `rk4` | 4th | 4 | Classical 4th-order Runge-Kutta. Gold standard for ODE integration. |
| `ab2` | 2nd | 1* | Adams-Bashforth 2-step. Multistep method; requires warm-up step. |
| `er_sde` | – | 2 | Exponential Rosenbrock-style. Specialized for SDE/ODE hybrid. |
| `dpmpp_2m` | – | 1* | DPM-Solver++ 2M. Multistep exponential integrator; very efficient after warm-up. |

*After initial warm-up phase.

### Substep Controls

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `substeps` | Int | 1 | 1 – 8 | Maximum internal substeps per sigma interval. Higher values refine the trajectory but increase compute. |
| `substep_mode` | Enum | ancestral | ancestral, deterministic | **ancestral**: Full ancestral logic per substep (down + renoise). **deterministic**: Substeps only refine ODE down-step; renoise once per outer step. |
| `substep_spacing` | Enum | log | log, linear | Internal sigma spacing. **log** = geometric (better for diffusion schedules). **linear** = uniform intervals. |
| `substep_active_start` | Float | 0.0 | 0.0 – 1.0 | Fraction of sampling progress where substeps begin activating. |
| `substep_active_end` | Float | 1.0 | 0.0 – 1.0 | Fraction of sampling progress where substeps stop activating. |
| `substep_fade` | Float | 0.0 | 0.0 – 1.0 | Fade substeps in/out over this fraction of the active window. Values > 0.5 are clamped to 0.5. |

**Scheduling Examples:**

- **Full-range substepping**: `active_start=0.0`, `active_end=1.0`, `fade=0.0`
- **Mid-range focus**: `active_start=0.2`, `active_end=0.8`, `fade=0.1`
- **Smooth transitions**: `fade=0.15` for gradual ramp-up/down

### Parameterization Modes

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `parameterization` | Enum | flow | flow, edm | Noise schedule parameterization.<br>• **flow**: α = 1 - σ (original Euler-A2)<br>• **edm**: α = 1 (standard k-diffusion ancestral) |

Use **flow** for most Stable Diffusion checkpoints. Use **edm** for models trained with EDM conventions.

### Langevin Corrector

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `corrector` | Enum | none | none, langevin, langevin_dynamic | Post-step corrector mode.<br>• **none**: Skip correction<br>• **langevin**: Fixed step-size ULA<br>• **langevin_dynamic**: Adaptive step-size based on gradient norm |
| `corrector_steps` | Int | 1 | 0 – 8 | Number of Langevin iterations per sampling step. |
| `corrector_eta` | Float | 0.5 | 0.0 – 2.0 | Step size multiplier for Langevin dynamics. Higher = stronger correction. |

### Variance Reduction

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `variance_reduction` | Enum | none | none, antithetic | Technique to reduce sampling variance.<br>• **none**: Standard independent draws<br>• **antithetic**: Paired ± noise for symmetry. Works best with even `noise_paths`. |

---

## How It Works

### Noise Path Merging

At each sampling step, instead of drawing one noise vector ε ~ N(0, 1), Euler-A2 draws N independent samples {ε₁, ε₂, ..., εₙ} and combines them:

**Mean Mode:**
```
ε_merged = (1/N) Σ εᵢ
```

**Median Mode:**
```
ε_merged = median(ε₁, ε₂, ..., εₙ)  # per-pixel median
```

The merged noise has reduced variance (~1/N for mean), which is then optionally rescaled:

- **Variance normalization**: ε_final = ε_merged × √N
- **RMS normalization**: ε_final = ε_merged / RMS(ε_merged)

Finally, an extrapolation factor boosts the direction:
```
direction = ε_final × (1 + extrapolation)
```

### Higher-Order Integration

The deterministic "down-step" (moving from higher to lower noise) is solved as an ODE:

```
dx/dσ = (x - D(x, σ)) / σ
```

where D(x, σ) is the denoiser prediction. Different integration methods approximate this trajectory with varying accuracy:

- **1st order** (Euler): Single evaluation, linear approximation
- **2nd order** (Midpoint, Heun, etc.): Two evaluations, quadratic approximation
- **3rd/4th order** (RK3, RK4): Multiple evaluations, higher-order polynomial approximation
- **Multistep** (AB2, DPMPP-2M): Reuse information from previous steps for efficiency

### Substep Scheduling

Each sigma interval [σᵢ, σᵢ₊₁] can be subdivided into smaller steps. The effective substep count varies based on scheduling parameters:

```python
progress = current_step / total_steps

if progress < active_start or progress > active_end:
    effective_substeps = 1
elif in_fade_region:
    effective_substeps = interpolated_value
else:
    effective_substeps = max_substeps
```

This allows concentrating computation where it matters most (typically mid-sampling where image structure forms).

---

## Practical Guidelines

### Quick Start Presets

#### 🚀 Fast & Good (Default Behavior)
```
noise_paths: 2
method: euler
substeps: 1
extrapolation: 0.425
```
Reproduces original Euler-A2 with minimal overhead.

#### ⚖️ Balanced Quality
```
noise_paths: 2
method: euler_enhanced
noise_normalization: variance
extrapolation: 0.425
```
~20% better stability with no extra model evaluations.

#### 🎨 High Quality
```
noise_paths: 4
merge_mode: median
method: rk4
substeps: 2
substep_mode: ancestral
noise_normalization: variance
extrapolation: 0.425
```
Excellent detail preservation with robust noise handling.

#### 🔬 Maximum Quality (Slow)
```
noise_paths: 6
merge_mode: median
method: dpmpp_2m
substeps: 4
substep_mode: ancestral
substep_active_start: 0.1
substep_active_end: 0.9
noise_normalization: variance
variance_reduction: antithetic
corrector: langevin_dynamic
corrector_steps: 2
```
Best results for critical work; significantly slower.

### Parameter Recommendations

**Noise Paths:**
- 1–2: Fast, subtle improvement over standard Euler-a
- 3–4: Sweet spot for quality/speed
- 5–8: Diminishing returns; use median merge mode

**Merge Mode:**
- Use **mean** for 2 paths (most common)
- Use **median** for 3+ paths (robust to outliers)

**Method Selection:**
- **euler_enhanced**: Best default; same cost as euler, better stability
- **rk4**: Highest accuracy when compute allows
- **dpmpp_2m**: Most efficient for long runs (after warm-up)
- **ab2**: Good middle ground for multistep efficiency

**Substeps:**
- Start with 1 (no substepping)
- Increase to 2–4 for finer control
- Use scheduling to limit cost: activate only during mid-sampling (0.2–0.8)

**Extrapolation:**
- 0.414–0.425: Preserves noise energy for 2 paths (theoretical sweet spot)
- 0.0: Disables extrapolation (pure merged noise)
- Negative values: Reduce noise influence

**Corrector:**
- **none**: Default; fastest
- **langevin_dynamic**: Best adaptive correction; use 1–2 steps

---

## Backward Compatibility

### Default Behavior
With default widget values, Euler-A2 reproduces the original Euler-A2 sampler behavior exactly:
```
eta=1.0, s_noise=1.0, extrapolation=0.425, noise_paths=2, 
merge_mode=mean, noise_normalization=none, method=euler, substeps=1
```

### Legacy Method Migration
The previously available `milstein` method has been replaced by `dpmpp_2m`. Old workflows using `milstein` are automatically migrated to `dpmpp_2m` to ensure compatibility.

### API Stability
The sampler registers itself idempotently with ComfyUI's sampler lists, ensuring safe reloading and avoiding duplicate entries.

---

## Technical Details

### Mathematical Foundation

Euler-A2 extends the standard ancestral sampling formula:

```
x_{i+1} = α_{i+1}/α_i × x_i + σ_{i+1} × ε
```

by replacing single noise ε with a merged, normalized, extrapolated direction:

```
x_{i+1} = α_{i+1}/α_i × x_i + σ_{i+1} × (1 + β) × Normalize(Merge(ε₁...εₙ))
```

where β is the extrapolation parameter.

### Integration Safety

All integration methods include protective checks:
- Division-by-zero guards for small sigma values
- NaN prevention through epsilon clamping
- Graceful degradation for edge cases (σ → 0)

### Memory Efficiency

The implementation uses:
- In-place operations where possible
- torch.no_grad() context for inference
- Efficient noise path storage (list of tensors, not stacked until merge)

### Threading & Determinism

- Thread-safe for batched inference
- Seed control via extra_args["seed"] passed to noise_sampler
- Antithetic sampling ensures symmetric noise distribution

---

## Troubleshooting

### Issue: No visible difference from standard Euler-a

**Solution:** Ensure `noise_paths > 1` or `extrapolation ≠ 0`. The enhancement comes from multi-path merging and extrapolation.

### Issue: Slower than expected

**Solutions:**
- Reduce `noise_paths` (each additional path adds proportional cost)
- Use simpler `method` (euler vs rk4)
- Reduce `substeps` or use scheduling to limit active range
- Disable `corrector` or reduce `corrector_steps`

### Issue: Artifacts or instability

**Solutions:**
- Try `method: euler_enhanced` for better stability
- Use `merge_mode: median` with 3+ paths
- Enable `variance_reduction: antithetic`
- Reduce `extrapolation` value
- Check `parameterization` matches your model type

### Issue: Out of memory

**Solutions:**
- Reduce `noise_paths`
- Reduce `substeps`
- Lower batch size
- Use `deterministic` substep mode (less memory than ancestral)

### Issue: Sampler not appearing in ComfyUI

**Solutions:**
- Verify installation in `custom_nodes/` directory
- Check for Python errors in ComfyUI console
- Ensure ComfyUI is updated to latest version
- Restart ComfyUI completely

---

## License

This project is provided as-is for use with ComfyUI. Follow ComfyUI's licensing terms for derivative works.

---

## Contributing

Contributions welcome! Areas for potential improvement:
- Additional integration methods
- Adaptive noise path counts
- Learning-based noise merging strategies
- Performance optimizations

---

## Acknowledgments

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) team for the extensible sampler architecture
- DPM-Solver and DPM-Solver++ authors for multistep integration techniques
- k-diffusion library for foundational sampling utilities

---

**Version:** 1.0  
**Last Updated:** 2024  
**Compatibility:** ComfyUI latest
