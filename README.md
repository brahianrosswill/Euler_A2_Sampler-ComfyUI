# Euler A2 Sampler for ComfyUI

<img width="2509" height="1698" alt="image" src="https://github.com/user-attachments/assets/16d24fcc-f4ea-4f2f-acdc-e800286f958c" />

<img width="620" height="567" alt="image" src="https://github.com/user-attachments/assets/bb8f0502-61a5-45b2-9205-276107cc38f1" />

## Overview

**Euler A2** is an advanced Euler-ancestral sampler variant for ComfyUI that stabilizes per-step noise injection by drawing multiple independent noise paths, merging them into a single direction, and optionally re-amplifying that direction. Think of it as "ancestral sampling with a committee vote on which way the noise should point."

## Features

- **Multi-Path Noise Merging**: Draw N independent noise paths at every step, merge them (mean or median) for a stabilized noise direction
- **Noise Normalization**: Optional variance or RMS normalization to control noise energy
- **Extrapolation Control**: Re-amplify the merged direction to restore or adjust noise magnitude
- **Higher-Order ODE Integration**: Support for multiple integration methods:
  - `euler` (1st order)
  - `midpoint`, `ralston`, `heun`, `dpm2` (2nd order)
  - `rk3` (3rd order)
  - `rk4` (4th order)
  - `ab2` (Adams-Bashforth 2-step, multistep method)
- **Internal Substepping**: Refine sigma intervals with configurable substeps in two modes:
  - `ancestral`: Full down-step + renoise per substep (finer SDE discretization)
  - `deterministic`: Substeps only refine the ODE down-step, renoise once per outer step
- **Active Range Control**: Restrict substepping to specific fractions of the sampling process with fade-in/out
- **Parameterization Support**: Both `flow` (α = 1 − σ, for flow-matching models like Flux, SD3) and `edm` (α = 1, for traditional SD 1.5/SDXL checkpoints)

## Installation

Clone or copy this repository into your ComfyUI custom nodes directory:

```bash
cd ComfyUI/custom_nodes
git clone <repository-url> euler_a2_sampler
```

Restart ComfyUI to load the new sampler node.

## Usage

The node appears in ComfyUI under: **sampling → custom_sampling → samplers → Euler A2 Sampler**

### Quick Start Presets

#### Smooth & Stable (Default)
Best for general use with flow-matching models.

| Parameter | Value |
|-----------|-------|
| eta | 1.0 |
| extrapolation | 0.425 |
| noise_paths | 2 |
| merge_mode | mean |
| noise_normalization | none |
| method | euler |
| parameterization | flow |

#### Ultra-Coherent
Best for portraits, architecture, and structure-critical images.

| Parameter | Value |
|-----------|-------|
| eta | 1.0 |
| extrapolation | 0.0 |
| noise_paths | 4 |
| merge_mode | median |
| noise_normalization | rms |
| active_end | 0.7 |
| method | heun |

#### EDM Standard
Best for SD 1.5, SDXL, Pony (traditional EDM diffusion models).

| Parameter | Value |
|-----------|-------|
| eta | 1.0 |
| extrapolation | 0.0 |
| noise_paths | 3 |
| merge_mode | mean |
| noise_normalization | variance |
| parameterization | **edm** |

## Parameters

### Core Ancestral Controls

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `eta` | float | 1.0 | 0.0–100.0 | Controls ancestral stochasticity (0=deterministic, 1=full ancestral) |
| `s_noise` | float | 1.0 | 0.0–100.0 | Global multiplier on injected noise magnitude |
| `extrapolation` | float | 0.425 | -10.0–10.0 | Boost factor for merged noise direction (gain = 1 + extrapolation) |

### Noise Path Merging

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `noise_paths` | int | 2 | 1–8 | Number of independent noise vectors drawn per step |
| `merge_mode` | choice | mean | mean, median | How noise paths are combined |
| `noise_normalization` | choice | none | none, variance, rms | Rescaling applied to merged noise |

### Active Range

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `active_start` | float | 0.0 | 0.0–1.0 | Fraction where noise merging begins |
| `active_end` | float | 1.0 | 0.0–1.0 | Fraction where noise merging ends |

### ODE Integration

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `method` | choice | euler | euler, midpoint, ralston, heun, dpm2, rk3, rk4, ab2 | ODE integration method |

### Internal Substepping

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `substeps` | int | 1 | 1–8 | Maximum internal subdivisions per sigma interval |
| `substep_mode` | choice | ancestral | ancestral, deterministic | Whether substeps also renoise |
| `substep_spacing` | choice | log | log, linear | Spacing for substep sigmas |
| `substep_active_start` | float | 0.0 | 0.0–1.0 | Fraction where substepping begins |
| `substep_active_end` | float | 1.0 | 0.0–1.0 | Fraction where substepping ends |
| `substep_fade` | float | 0.0 | 0.0–0.5 | Fraction of active window for ramping substeps |

### Parameterization

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `parameterization` | choice | flow | flow, edm | Model compatibility mode |

## When to Use Euler A2

- You want **smoother, more coherent images** than standard Euler-a produces
- Standard ancestral sampling feels too "noisy" or chaotic at high CFG
- You want to experiment with **higher-order integration methods** inside an ancestral framework
- You want **substepping only during specific phases** of sampling to save compute

## Documentation

For detailed parameter explanations, interaction matrices, and troubleshooting, see [EulerA2_Parameter_GuideV3.md](./EulerA2_Parameter_GuideV3.md).

## License

Same license as ComfyUI.
