# Euler A2 Sampler — Complete Practical Parameter Guide

> **Node:** `Euler_A2_Sampler`  
> **Display Name:** Euler A2 Sampler  
> **Category:** `sampling/custom_sampling/samplers`  

---

## Table of Contents

1. [What Is This Sampler?](#what-is-this-sampler)
2. [Quick-Start Presets](#quick-start-presets)
3. [Parameter Reference](#parameter-reference)
   - [Core Ancestral Controls](#core-ancestral-controls)
   - [Noise-Path Merging](#noise-path-merging)
   - [Active Range](#active-range)
   - [ODE Integration Method](#ode-integration-method)
   - [Internal Substepping](#internal-substepping)
   - [Parameterization](#parameterization)
4. [Parameter Interaction Matrix](#parameter-interaction-matrix)
5. [Performance & NFE Budget](#performance--nfe-budget)
6. [Troubleshooting](#troubleshooting)
7. [Version History](#version-history)

---

## What Is This Sampler?

Euler A2 is an **Euler-ancestral variant** that stabilizes the per-step noise injection by drawing multiple independent noise paths, merging them into a single direction, and optionally re-amplifying that direction. Think of it as "ancestral sampling with a committee vote on which way the noise should point."

### Core idea
1. At each step, sample **N noise directions** instead of 1.
2. **Merge** them (mean or median) → a smoother, less erratic direction.
3. **Normalize** the merged direction if desired.
4. **Extrapolate** along it (boost the magnitude).
5. Optionally use **higher-order ODE integrators** and **internal substeps** for the deterministic down-step.

### When to use it
- You want **smoother, more coherent images** than standard Euler-a produces.
- You find normal ancestral sampling too "noisy" or chaotic at high CFG.
- You want to experiment with **higher-order methods** (Heun, RK4, DPM2) inside an ancestral framework.
- You want **substepping only in the middle of sampling** (e.g., only while structure forms) to save compute.

---

## Quick-Start Presets

Copy these into your workflow as starting points, then tune.

### Preset A: "Smooth & Stable" (Default)
Best for: general use, flow-matching models, avoiding chaos.

| Parameter | Value | Why |
|-----------|-------|-----|
| `eta` | `1.0` | Full ancestral |
| `s_noise` | `1.0` | Standard noise strength |
| `extrapolation` | `0.425` | Restores energy for 2 merged paths |
| `noise_paths` | `2` | Good balance of smoothness vs speed |
| `merge_mode` | `mean` | Standard averaging |
| `noise_normalization` | `none` | Let extrapolation handle energy |
| `method` | `euler` | Fast, 1 eval per substep |
| `substeps` | `1` | No internal substeps |
| `parameterization` | `flow` | For flow-matching models (Flux, SD3, etc.) |

### Preset B: "Ultra-Coherent"
Best for: portraits, architecture, anything where structure matters more than texture randomness.

| Parameter | Value | Why |
|-----------|-------|-----|
| `eta` | `1.0` | Full ancestral |
| `s_noise` | `1.0` | |
| `extrapolation` | `0.0` | With normalization, no boost needed |
| `noise_paths` | `4` | Very stable direction |
| `merge_mode` | `median` | Robust to outlier noise draws |
| `noise_normalization` | `rms` | Locks noise energy to exactly 1 every step |
| `active_start` | `0.0` | |
| `active_end` | `0.7` | Stop merging in last 30% — let fine detail stay stochastic |
| `method` | `heun` | 2nd order, better trajectory |
| `substeps` | `1` | |
| `parameterization` | `flow` or `edm` | Match your model |

### Preset C: "EDM Standard"
Best for: SD 1.5, SDXL, Pony, etc. (traditional EDM diffusion models).

| Parameter | Value | Why |
|-----------|-------|-----|
| `eta` | `1.0` | |
| `s_noise` | `1.0` | |
| `extrapolation` | `0.0` | EDM ancestral already has correct energy |
| `noise_paths` | `3` | |
| `merge_mode` | `mean` | |
| `noise_normalization` | `variance` | Restores unit variance (×√3) |
| `method` | `euler` | |
| `substeps` | `1` | |
| `parameterization` | **`edm`** | **Critical** — uses α=1 math compatible with EDM |

### Preset D: "Quality at Cost"
Best for: final renders where you have time.

| Parameter | Value | Why |
|-----------|-------|-----|
| `eta` | `1.0` | |
| `s_noise` | `1.0` | |
| `extrapolation` | `0.0` | |
| `noise_paths` | `3` | |
| `noise_normalization` | `rms` | |
| `method` | `rk4` | 4th order, most accurate trajectory |
| `substeps` | `4` | Fine internal discretization |
| `substep_mode` | `deterministic` | Substeps refine ODE only; renoise once per step |
| `substep_active_start` | `0.2` | Skip early high-noise steps |
| `substep_active_end` | `0.6` | Stop after structure is locked |
| `substep_fade` | `0.25` | Smooth ramp in/out |
| `parameterization` | match model | |

---

## Parameter Reference

---

### Core Ancestral Controls

#### `eta` (FLOAT, default: 1.0, range: 0.0 – 100.0)

**What it does:** Controls how "ancestral" the step is — i.e., how much the sampler deviates from the deterministic probability-flow ODE toward the fully stochastic SDE.

| Value | Behaviour |
|-------|-----------|
| `0.0` | **Deterministic.** No noise injection. Equivalent to DDIM / probability-flow ODE. |
| `0.5` | Halfway. Some noise, but trajectory stays closer to the ODE. |
| `1.0` | **Full ancestral.** Standard Euler-ancestral behaviour. Maximum stochasticity per step. |
| `> 1.0` | **Hyper-ancestral.** Injects *more* noise than the variance schedule technically allows. Can produce interesting textures but may diverge. |

**Practical tips:**
- Start at `1.0`. Reduce to `0.7`–`0.9` if images feel too noisy or unfocused.
- At `eta = 0`, `noise_paths`, `extrapolation`, and `noise_normalization` have **no effect** (no noise is injected).
- If you use `eta > 1.0`, consider lowering `s_noise` to compensate.

**Interaction:** Directly multiplies against `s_noise`. If either is 0, the sampler becomes deterministic regardless of the other.

---

#### `s_noise` (FLOAT, default: 1.0, range: 0.0 – 100.0)

**What it does:** Global multiplier on the magnitude of injected noise at every step.

| Value | Behaviour |
|-------|-----------|
| `0.0` | No noise. Deterministic, regardless of `eta`. |
| `0.5` | Half the normal noise. Images are smoother but may look "washed out" or overly converged. |
| `1.0` | Standard ancestral noise level. |
| `1.2`–`1.5` | More texture, more variation between seeds. Good for artistic / abstract work. |
| `> 2.0` | Usually too much. Images break into noise unless `eta` is very low. |

**Practical tips:**
- This is your "texture volume" knob. Turn it down for clean outputs, up for gritty / painterly results.
- If you increase `noise_paths` and use `noise_normalization = "variance"`, the merged direction already has restored energy, so you may want `s_noise = 1.0` or slightly less.
- In the EDM parameterization, `s_noise` scales the raw `sigma_up` from `get_ancestral_step`.

---

#### `extrapolation` (FLOAT, default: 0.425, range: -10.0 – 10.0)

**What it does:** After merging N noise paths, the sampler can boost the merged direction beyond its natural magnitude. Total noise gain = `1.0 + extrapolation`.

| Value | Behaviour |
|-------|-----------|
| `-1.0` | **Zero gain.** The merged direction is completely cancelled. No noise is added even if `eta > 0`. |
| `0.0` | **Unit gain.** Use exactly the merged direction as-is. Best when combined with `noise_normalization = "variance"` or `"rms"`. |
| `0.414` (~√2 − 1) | Theoretically preserves total noise energy for 2 unnormalized mean-merged paths. |
| `0.425` (default) | Slightly above √2−1. Compensates for the slight energy loss from merging. |
| `> 1.0` | Aggressive boost. Can create strong directional noise artifacts — interesting for stylization. |
| `< 0.0` | Attenuates the noise. Values between `-0.5` and `0.0` give "soft ancestral" results. |

**Practical tips:**
- With `noise_normalization = "none"` and `noise_paths = 2`, keep `extrapolation ≈ 0.414–0.5`.
- With `noise_normalization = "variance"` or `"rms"`, set `extrapolation = 0.0`. The normalization already restores energy; extra boost will overdrive.
- With `noise_paths = 1`, `extrapolation` still applies — you're just boosting a single noise draw. Usually leave at `0.0` unless you want exaggerated noise.

**Math note:** For `N` independent standard-normal draws, the mean has variance `1/N`. To restore unit variance you need gain `√N`. The default `0.425` with `N=2` gives gain `1.425`, which is close to `√2 ≈ 1.414`.

---

### Noise-Path Merging

#### `noise_paths` (INT, default: 2, range: 1 – 8)

**What it does:** Number of independent noise vectors drawn and merged per step.

| Value | Behaviour |
|-------|-----------|
| `1` | No merging. Standard single-noise ancestral step. Fastest. |
| `2` | Default. Two paths averaged → variance halved. Good smoothness / speed tradeoff. |
| `3` | Smoother. Works well with `merge_mode = "median"` (odd count = true median). |
| `4`–`5` | Very stable. Diminishing returns start here. |
| `6`–`8` | Overkill for most use cases. Use only with `merge_mode = "median"` and if you have GPU headroom. |

**Practical tips:**
- Each path is a full `randn_like(x)` call, so memory and RNG overhead scale linearly. On large latents (e.g. 1024×1024×16) this is noticeable.
- `noise_paths = 2` with `extrapolation = 0.425` is the sweet spot for most workflows.
- If you want the **smoothest possible** result, use `noise_paths = 5`, `merge_mode = "median"`, `noise_normalization = "rms"`, `extrapolation = 0.0`.

---

#### `merge_mode` (choice: `mean` | `median`, default: `mean`)

**What it does:** How the N noise draws are combined into one direction.

| Mode | Behaviour | Best For |
|------|-----------|----------|
| `mean` | Simple average. Variance drops as `1/N`. Fast, differentiable, predictable. | General use, even N counts. |
| `median` | Per-pixel median across the N draws. Completely ignores outlier pixels. | `noise_paths ≥ 3`, removing rare extreme noise spikes. Slightly more coherent but can look "cleaner" than mean. |

**Practical tips:**
- `median` with `noise_paths = 2` uses the lower of the two values per pixel (torch.median behavior for even N). This is usually **not** what you want. Use `median` with `3` or more paths.
- `mean` preserves Gaussian statistics better; `median` produces a slightly non-Gaussian noise distribution which can subtly sharpen edges.
- If switching to `median`, try increasing `noise_paths` by 1 and lowering `extrapolation` by ~0.1.

---

#### `noise_normalization` (choice: `none` | `variance` | `rms`, default: `none`)

**What it does:** Rescales the merged noise direction before injection.

| Mode | Formula | Effect |
|------|---------|--------|
| `none` | — | Merged direction used as-is. With `N` paths and `mean`, variance is `1/N`. |
| `variance` | `× √N` | Statistically restores unit variance. Theoretically correct for i.i.d. Gaussian draws. |
| `rms` | `÷ RMS(noise)` | Empirically forces the per-sample RMS to exactly 1.0. Removes **all** energy fluctuation between steps and seeds. Most deterministic feel. |

**Practical tips:**
- **`none`** → pair with `extrapolation ≈ 0.4–0.5` (for N=2).
- **`variance`** → pair with `extrapolation = 0.0`. The math already handles energy.
- **`rms`** → pair with `extrapolation = 0.0`. Gives the most "controlled" ancestral feel. Every step gets identical noise energy, so variation comes purely from direction, not magnitude.
- `rms` is computationally cheap (one `mean` and `sqrt` per sample in batch).
- If your latent is all zeros (rare, but can happen with masked inpainting), `rms` mode has a guard to avoid division-by-near-zero.

---

### Active Range

#### `active_start` (FLOAT, default: 0.0, range: 0.0 – 1.0)
#### `active_end` (FLOAT, default: 1.0, range: 0.0 – 1.0)

**What they do:** Restrict the noise-path merging + extrapolation to a fraction of the sampling process. Outside this range, the sampler falls back to **standard single-noise ancestral** (as if `noise_paths = 1` and `extrapolation = 0`).

| Range | Effect |
|-------|--------|
| `[0.0, 1.0]` | Merging active on every step. |
| `[0.0, 0.6]` | Merging only while coarse structure forms. Last 40% of steps use normal noise for fine texture. |
| `[0.2, 0.8]` | Skip the noisiest early steps (where direction is nearly random anyway) and the final denoising steps. |
| `[0.5, 1.0]` | Only merge in late steps. Can help stabilize fine detail without over-smoothing early composition. |

**Practical tips:**
- Early steps (high σ) are dominated by noise; merging many paths there has limited benefit because all directions are nearly random. Consider `active_start = 0.1` or `0.2`.
- Late steps (low σ) control texture and fine detail. If you want to preserve "happy accidents" in hair, fabric, or backgrounds, set `active_end = 0.7` or `0.8`.
- These only affect the **noise merging**. The ODE integration (`method`, `substeps`) still runs on every step.

---

### ODE Integration Method

#### `method` (choice: `euler` | `midpoint` | `ralston` | `heun` | `dpm2` | `rk3` | `rk4`, default: `euler`)

**What it does:** Determines the numerical integrator used for the deterministic down-step (the probability-flow ODE `dx/dσ = (x − denoised)/σ`).

| Method | Order | Evaluations per segment | Character |
|--------|-------|------------------------|-----------|
| `euler` | 1st | 1 | Fast, least accurate. Good enough at typical step counts (20–30). |
| `midpoint` | 2nd | 2 | Standard RK2. Better accuracy than Euler, especially with fewer steps. |
| `ralston` | 2nd | 2 | Optimized RK2 with minimal truncation error bound. Very slightly better than midpoint in theory; often indistinguishable in practice. |
| `heun` | 2nd | 2 | Trapezoidal rule. Tends to be slightly more stable than midpoint on stiff trajectories. |
| `dpm2` | 2nd | 2 | DPM-Solver-2 single-step. Uses geometric midpoint and exponential integrator tailored to diffusion ODEs. Often the best 2nd-order choice for diffusion specifically. |
| `rk3` | 3rd | 3 | Kutta's 3rd order. Noticeably better accuracy than 2nd order. Use when you have substeps > 1 and want quality. |
| `rk4` | 4th | 4 | Classical RK4. Most accurate, most expensive. Best paired with `substeps` and `substep_mode = "deterministic"` to amortize cost. |

**Practical tips:**
- At **20+ steps**, `euler` is usually fine. The gains from higher-order methods diminish because the step size is already small.
- At **< 15 steps**, switch to `dpm2` or `heun`. The improved trajectory accuracy matters more.
- `rk4` is overkill unless you are also using `substeps > 1` or very few outer steps (< 10).
- `dpm2` is the "diffusion-native" 2nd-order choice. If you only upgrade one thing from Euler, make it `dpm2`.
- All methods fall back to Euler automatically when `sigma_to == 0` (the final step) because the ODE derivative is singular there.

---

### Internal Substepping

#### `substeps` (INT, default: 1, range: 1 – 8)

**What it does:** Maximum number of internal subdivisions of each outer sigma interval. Each substep calls the model at least once, so this multiplies your NFE (number of function evaluations).

| Value | Effect |
|-------|--------|
| `1` | No substeps. One model call per outer step (plus any from the integration method). |
| `2` | Each outer interval split in two. 2× model calls for that interval's down-step. |
| `4` | 4× model calls. Very fine discretization. |
| `8` | 8× model calls. Extreme refinement. |

**Practical tips:**
- Substeps are most beneficial in the **middle** of sampling (σ ≈ 1.0 → 0.3) where the ODE curvature is highest.
- Use `substep_active_start` / `substep_active_end` to avoid wasting compute on early/late steps.
- If `substep_mode = "ancestral"`, each substep also renoises, so you get a finer SDE discretization. This can make images slightly smoother but costs more.
- If `substep_mode = "deterministic"`, only the ODE is refined; renoise happens once per outer step. Better quality per NFE.

---

#### `substep_mode` (choice: `ancestral` | `deterministic`, default: `ancestral`)

| Mode | Behaviour | Use When |
|------|-----------|----------|
| `ancestral` | Every substep does a full ancestral segment: down-step + renoise. This is a finer SDE discretization. | You want smoother, more "dreamlike" transitions. You have GPU budget. |
| `deterministic` | Substeps only refine the ODE down-step. The renoise is applied **once** after all substeps complete. | You want better trajectory accuracy without extra stochasticity. More efficient. |

**Practical tips:**
- `deterministic` is almost always the better choice when `substeps > 2`. It gives you the accuracy benefit without the compounding noise variance of multiple renoise operations.
- `ancestral` with `substeps = 2` is a nice middle ground — slightly finer SDE without too much cost.

---

#### `substep_spacing` (choice: `log` | `linear`, default: `log`)

**What it does:** How the internal sigma points are spaced within each outer interval.

| Spacing | Formula | Best For |
|---------|---------|----------|
| `log` | Geometric: `σ_j = exp( log(σ_a) + j/count · (log(σ_b) − log(σ_a)) )` | Diffusion ODEs, where dynamics are naturally log-linear. **Recommended.** |
| `linear` | Uniform: `σ_j = σ_a + j/count · (σ_b − σ_a)` | When you specifically want equal spacing in sigma. Falls back automatically if `σ_b == 0`. |

**Practical tips:**
- `log` is the default for a reason: diffusion processes evolve on a log-sigma timescale. Use `linear` only if you have a specific reason to suspect the ODE is linear in σ (rare).
- When `σ_b == 0` (final step), log spacing is mathematically impossible; the code silently falls back to linear.

---

#### `substep_active_start` (FLOAT, default: 0.0, range: 0.0 – 1.0)
#### `substep_active_end` (FLOAT, default: 1.0, range: 0.0 – 1.0)

**What they do:** Restrict substeps to a fraction of the sampling process. Outside this range, `effective_substeps = 1` regardless of the `substeps` setting.

| Range | Typical Use |
|-------|-------------|
| `[0.0, 1.0]` | Substeps always active. |
| `[0.1, 0.7]` | Skip the noisiest early steps and the fine late steps. Focus compute where structure forms. |
| `[0.0, 0.5]` | Heavy refinement in first half, then let the model coast. Good for fast previews. |

---

#### `substep_fade` (FLOAT, default: 0.0, range: 0.0 – 1.0)

**What it does:** Ramps the number of substeps from 1 up to `substeps` (and back down) over a fraction of the active window. Creates a smooth transition instead of an abrupt on/off.

| Value | Behaviour |
|-------|-----------|
| `0.0` | Abrupt: substeps instantly go from 1 to N at `substep_active_start`. |
| `0.25` | Ramp up over the first 25% of the active window, ramp down over the last 25%. |
| `0.5` | Maximum symmetric ramp. The entire active window is ramp (never flat at N). |
| `> 0.5` | **Clamped to 0.5 internally** to prevent overlapping ramp-up/ramp-down windows. |

**Practical tips:**
- A fade of `0.15`–`0.25` usually looks smoother than abrupt switching.
- If `substep_active_start = 0.2`, `substep_active_end = 0.6`, and `substep_fade = 0.25`:
  - Ramp up: progress 0.2 → 0.3 (25% of 0.4 window)
  - Full substeps: progress 0.3 → 0.5
  - Ramp down: progress 0.5 → 0.6

---

### Parameterization

#### `parameterization` (choice: `flow` | `edm`, default: `flow`)

**What it does:** Tells the sampler which noise schedule / data parameterization the model was trained on. This changes the ancestral renoise math completely.

| Mode | Assumption | `α(σ)` | Compatible Models |
|------|-----------|--------|-------------------|
| `flow` | Flow-matching / linear interpolation | `1 − σ` | Flux, SD3, Stable Diffusion 3, some custom flow models. σ typically ranges [0, 1]. |
| `edm` | Karras EDM / traditional diffusion | `1` | SD 1.5, SDXL, Pony, realistic vision, most Stable Diffusion checkpoints. σ ranges up to ~14.6. |

**Why this matters:**
- In `flow` mode, the sampler uses `α = 1 − σ` to compute the renoise variance:  
  `base = (α_next / α_down) · x_det`  
  `renoise² = σ_next² − σ_down² · (α_next / α_down)²`
- In `edm` mode, the sampler uses k-diffusion's standard `get_ancestral_step`:  
  `σ_down, σ_up = get_ancestral_step(σ_from, σ_to, eta)`  
  `x = x_det + noise · s_noise · σ_up`

**Critical warning:**
- If you run an **EDM model** (SD 1.5, SDXL) with `parameterization = "flow"` and `σ > 1`, `α = 1 − σ` becomes **negative**. This flips the data estimate and produces **completely wrong noise variance**. Images will look corrupted or oversaturated.
- If you run a **flow model** with `parameterization = "edm"`, the noise energy will be slightly off (usually too low), producing slightly blurrier or less textured results.

**Practical tips:**
- **Flux / SD3** → `flow`
- **SD 1.5 / SDXL / Pony / anything with "sd" in the name** → `edm`
- When in doubt, try both. The wrong one is usually obvious within 5 steps (corrupted colors / extreme contrast).

---

## Parameter Interaction Matrix

| Parameter A | Parameter B | Interaction |
|-------------|-------------|-------------|
| `eta` | `s_noise` | If either is 0, sampler is deterministic. Both scale noise; `eta` controls the split point, `s_noise` scales the injection. |
| `noise_paths` | `extrapolation` | With `noise_normalization = "none"`, use `extrapolation ≈ √N − 1`. With normalization, use `extrapolation = 0`. |
| `noise_paths` | `merge_mode` | `median` needs `noise_paths ≥ 3` to be meaningful. With `N=2`, median is just `min(a,b)` per pixel. |
| `noise_normalization` | `extrapolation` | If normalization is active, `extrapolation` should usually be `0.0` or very small. Double-boosting causes over-noise. |
| `active_start` / `active_end` | `noise_paths` | Restricting the active range lets you use high `noise_paths` without slowing down the entire sampling process. |
| `method` | `substeps` | Higher-order methods benefit more from substeps. `euler` + `substeps=4` is often worse than `dpm2` + `substeps=2` for the same NFE. |
| `substep_mode` | `substeps` | `ancestral` multiplies noise variance by substeps count (compounding renoise). `deterministic` does not. Prefer `deterministic` when `substeps > 2`. |
| `substep_fade` | `substep_active_start/end` | Fade is computed as a fraction of the active window, not the total sampling process. |
| `parameterization` | `eta` | In `edm` mode, `eta` is passed directly to `get_ancestral_step`. In `flow` mode, `eta` scales a ratio directly. |

---

## Performance & NFE Budget

### Model evaluations per outer step

| Method | Evals per substep | With `substeps = N` |
|--------|-------------------|---------------------|
| `euler` | 1 | N |
| `midpoint` | 2 | 2N |
| `ralston` | 2 | 2N |
| `heun` | 2 | 2N |
| `dpm2` | 2 | 2N |
| `rk3` | 3 | 3N |
| `rk4` | 4 | 4N |

### Total NFE estimate

```
NFE ≈ (outer_steps − 1) × evals_per_substep × avg_effective_substeps
```

Where `avg_effective_substeps` depends on `substeps`, `substep_active_start/end`, and `substep_fade`.

**Example:** 20 outer steps, `method = dpm2` (2 evals), `substeps = 4`, active range `[0.2, 0.6]`, fade `0.25`.
- Active window = 40% of steps = 8 steps
- Ramp up/down = 2 steps each at ~2 substeps, 4 steps at 4 substeps
- Average substeps in window ≈ 3
- NFE ≈ 19 × 2 × 3 = **114 model evaluations**

Compare to standard Euler-a: 20 evaluations. You are trading **5.7× compute** for smoother trajectory.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| **Oversaturated / inverted colors** | `parameterization = "flow"` on an EDM model | Switch to `"edm"` |
| **Muddy / blurry textures** | `parameterization = "edm"` on a flow model | Switch to `"flow"` |
| **Image looks almost identical across seeds** | `eta = 0` or `s_noise = 0` | Increase both to `1.0` |
| **Image is pure noise / never converges** | `extrapolation` too high with `noise_normalization = "rms"` | Set `extrapolation = 0.0` |
| **Weird banding or step artifacts** | `substep_fade` overlap (old code) or abrupt substeps | Update to latest code; use `substep_fade = 0.25` |
| **Slower than expected** | `noise_paths` too high, or substeps active on all steps | Reduce `noise_paths` to 2; restrict `substep_active_start/end` |
| **No visible difference vs Euler-a** | `noise_paths = 1`, `extrapolation = 0`, `substeps = 1`, `method = euler` | This IS Euler-a. Enable merging or higher-order methods. |
| **Faces look plastic / too smooth** | `noise_paths` too high, or active range extends too late | Set `active_end = 0.6` or reduce `noise_paths` |
| **Checkerboard artifacts in background** | `median` merge with even `noise_paths` | Use odd `noise_paths` (3, 5) or switch to `mean` |

---

## Version History

| Version | Changes |
|---------|---------|
| Original | Initial release with flow parameterization only. |
| Refined | Added `parameterization` selector (`flow` / `edm`), fixed `substep_fade` overlap bug, added zero-RMS guard, ensured callback tensor types, added `ValueError` for unknown methods. |

---

*Document generated for Euler A2 Sampler — refined edition.*
