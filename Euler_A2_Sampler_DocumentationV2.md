# Euler A2 Sampler — Documentation

A custom ancestral sampler node for **ComfyUI**. It extends the classic Euler
ancestral sampler by drawing **multiple independent noise paths per step**,
merging them into a single stabilized noise direction, and extrapolating along
that direction. On top of that it offers **higher-order integration methods**
(Euler, midpoint, Heun, RK4) for the deterministic down-step and **internal
substepping** of every sigma interval. The result is ancestral sampling with
controllable smoothness, stability, noise energy, and integration accuracy.

- Node ID: `Euler_A2_Sampler` (display name: **Euler A2 Sampler**)
- Category: `sampling/custom_sampling/samplers`
- Sampler name registered in ComfyUI: `euler_a2`
- Output: `SAMPLER` (connect to any node that accepts a sampler, e.g. **SamplerCustom**)

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Integration Methods](#integration-methods)
3. [Substeps](#substeps)
4. [Installation](#installation)
5. [Node Reference](#node-reference)
6. [Parameter Guide](#parameter-guide)
7. [Recommended Recipes](#recommended-recipes)
8. [Behavior & Compatibility Notes](#behavior--compatibility-notes)
9. [Technical Details](#technical-details)
10. [FAQ](#faq)

---

## How It Works

Each step performs the following (all math uses the `alpha = 1 - sigma`
parametrization the sampler was designed for):

1. **Model evaluation** — the model predicts `denoised` from the current
   latent `x` at `sigma_i`.

2. **Ancestral down-step** — an intermediate sigma is computed, interpolating
   between a fully deterministic step (`eta = 0`) and a fully ancestral step
   (`eta = 1`):

   ```
   downstep_ratio = 1 + (sigma_{i+1} / sigma_i - 1) * eta
   sigma_down     = clamp(sigma_{i+1} * downstep_ratio, 0, sigma_i)
   ```

   The latent is then integrated from `sigma_i` to `sigma_down` along the
   probability-flow ODE `dx/dσ = (x − denoised)/σ`, using the selected
   integration **method** (default `euler`, which reproduces the classic
   formula `x_det = r·x + (1−r)·denoised` with `r = sigma_down / sigma_i`).

3. **Renoise base** — the path is rescaled to the target alpha level, and the
   exact noise magnitude needed to reach `sigma_{i+1}` is derived:

   ```
   alpha_{i+1} = 1 - sigma_{i+1}
   alpha_down  = max(1 - sigma_down, eps)
   base        = (alpha_{i+1} / alpha_down) * x_det
   noise_scale = s_noise * sqrt(max(sigma_{i+1}^2 - sigma_down^2 * (alpha_{i+1} / alpha_down)^2, 0))
   ```

4. **Multi-path noise merging** — instead of injecting one fresh noise draw,
   `noise_paths` independent draws are merged into a single direction
   (`mean` or per-pixel `median`), optionally re-normalized:

   ```
   direction = normalize(merge(noise_1, ..., noise_N))
   x = base + (1 + extrapolation) * direction * noise_scale
   ```

**Why this helps:** a mean of N independent noise draws has variance ~1/N, so
the injected noise direction is far more stable from step to step — less
random drift, smoother composition — while `extrapolation` re-amplifies that
direction to restore (or deliberately over/under-shoot) the original noise
energy. For N = 2 with no normalization, the energy-preserving gain is
√2 ≈ 1.4142, i.e. `extrapolation = √2 − 1 ≈ 0.4142`; the default `0.425`
matches the original Euler A2 behavior (≈ 1.5 % extra energy).

**Active range:** merging + extrapolation can be restricted to a fraction of
the step range (`active_start` … `active_end`). Outside that window a plain
single-noise ancestral step is used, preserving fine-grained ancestral detail
where you want it.

---

## Integration Methods

The `method` parameter selects the numerical order of the deterministic
down-step (the ODE integration from `sigma_i` to `sigma_down`):

| Method | Order | Model evals per (sub)step | Notes |
|---|---|---|---|
| `euler` | 1st | 1 | Classic formula; exact reproduction of the original sampler. |
| `midpoint` | 2nd | 2 | Evaluates the derivative at the interval midpoint. |
| `heun` | 2nd | 2 | Trapezoidal average of start/end derivatives; the end prediction is reused as the next substep's start prediction (saves one eval when chaining). |
| `rk4` | 4th | 4 | Classical Runge–Kutta; highest accuracy per step. |

Higher order buys integration accuracy at the cost of extra model evaluations
(NFE). It is most visible at low step counts or with `eta = 0`
(fully deterministic sampling, where the ODE error is the whole error).
Methods verified to converge at (better than) their nominal order.

Fallbacks: segments ending at `sigma = 0` always use Euler (the ODE derivative
is undefined at 0), and the final step of a schedule short-circuits to
`x = denoised` as before.

---

## Substeps

`substeps` splits every sigma interval `[sigma_i, sigma_{i+1}]` into N
internal segments. Two modes control what happens per segment:

| Mode | Behavior | Use when |
|---|---|---|
| `ancestral` (default) | Each substep performs the full down-step **and** renoise — a finer discretization of the same SDE. | You want a denser ancestral trajectory (more, smaller noise injections). With matching sigma points this is mathematically identical to running on a finer schedule (verified to float precision). |
| `deterministic` | Substeps only refine the ODE down-step; the renoise to `sigma_{i+1}` happens **once** per outer step. | You want higher integration accuracy without changing the noise structure of the trajectory. |

`substep_spacing` controls the internal sigma points:

- `log` (default) — geometric spacing (uniform in log-σ), the natural
  coordinate for diffusion schedules. Falls back to `linear` when the target
  sigma is 0.
- `linear` — uniform spacing in σ.

**Cost:** NFE per outer step ≈ `substeps × evals(method)` for the deterministic
part (plus one eval per substep boundary in `ancestral` mode; Heun/RK4 reuse
their end evaluation across boundaries). Keep `substeps × method-evals`
modest (e.g. RK4 × 4 substeps ≈ 16 evals per sampling step).

With `eta = 0` / `s_noise = 0` (deterministic sampling), substeps refine the
integration in both modes.

---

## Installation

1. Copy `nodes.py` into your custom node package folder, e.g.
   `ComfyUI/custom_nodes/<your-pack>/nodes.py`, and make sure the package's
   `__init__.py` imports it.
2. Restart ComfyUI.
3. Add the node via **Add Node → sampling → custom_sampling → samplers →
   Euler A2 Sampler**, and connect its `SAMPLER` output to a sampler consumer
   (e.g. **SamplerCustom**).

On import, the module registers `euler_a2` into ComfyUI's sampler lists
(`KSAMPLER_NAMES`, `SAMPLER_NAMES`, `KSampler.SAMPLERS`) and exposes the
sampling function as `comfy.k_diffusion.sampling.sample_euler_a2`, so
`euler_a2` is also selectable anywhere a built-in sampler name is accepted.

---

## Node Reference

### Required inputs

| Input | Type | Default | Range | Description |
|---|---|---|---|---|
| `eta` | FLOAT | `1.0` | 0.0 – 100.0 | Ancestral interpolation. `0` = deterministic (DDIM-like), `1` = full ancestral. |
| `s_noise` | FLOAT | `1.0` | 0.0 – 100.0 | Global multiplier on the injected noise magnitude. |
| `extrapolation` | FLOAT | `0.425` | −10.0 – 10.0 | Extra gain along the merged noise direction. Total gain = `1 + extrapolation`. Energy-preserving value: `√N − 1` without normalization, `0` with normalization. |

### Optional inputs

| Input | Type | Default | Range / Choices | Description |
|---|---|---|---|---|
| `noise_paths` | INT | `2` | 1 – 8 | Number of independent noise draws merged per step. More paths = smoother, more stable direction (variance ~1/N for a mean). |
| `merge_mode` | COMBO | `mean` | `mean`, `median` | How the noise paths are combined. `median` is robust to outlier draws; use 3+ paths (ideally odd). |
| `noise_normalization` | COMBO | `none` | `none`, `variance`, `rms` | `variance` rescales the merged noise by √N (statistical unit-variance correction). `rms` forces per-sample RMS to exactly 1 every step (removes energy fluctuations between steps and seeds). |
| `active_start` | FLOAT | `0.0` | 0.0 – 1.0 | Fraction of the step range where merging + extrapolation begins (0 = first step). |
| `active_end` | FLOAT | `1.0` | 0.0 – 1.0 | Fraction of the step range where merging + extrapolation ends (1 = last step). |
| `method` | COMBO | `euler` | `euler`, `midpoint`, `heun`, `rk4` | Integration order of the deterministic down-step (1st / 2nd / 2nd / 4th order). See [Integration Methods](#integration-methods). |
| `substeps` | INT | `1` | 1 – 8 | Internal substeps per sigma interval; multiplies model evaluations. |
| `substep_mode` | COMBO | `ancestral` | `ancestral`, `deterministic` | Whether each substep renoises (finer SDE) or only refines the ODE down-step (renoise once). See [Substeps](#substeps). |
| `substep_spacing` | COMBO | `log` | `log`, `linear` | Internal sigma spacing: geometric (`log`) or uniform (`linear`). |

### Outputs

| Output | Type | Description |
|---|---|---|
| `SAMPLER` | SAMPLER | Configured `euler_a2` sampler instance, ready for SamplerCustom or any SAMPLER input. |

---

## Parameter Guide

### `eta`
Controls how stochastic each step is. `1.0` is standard ancestral behavior.
Lower values shrink the renoise amount; `0` disables noise injection entirely
(pure deterministic ODE path — combine with `method`/`substeps` for accuracy).

### `s_noise`
Scales the injected noise. `1.0` = exact ancestral amount. Values below 1
under-noise (slightly smoother, less diversity); above 1 over-noise.

### `extrapolation`
The signature parameter. The merged noise direction is applied with total gain
`1 + extrapolation`:

- `0` → merged noise applied at `noise_scale` (with normalization, this is
  exactly energy-preserving).
- `0.425` (default) → classic A2 behavior for 2 unaveraged paths
  (≈ energy-preserving, since `(1.425)² / 2 ≈ 1.015`).
- Negative values → suppress the merged direction (`−1` removes it entirely,
  leaving the deterministic base).
- Values above the energy-preserving point → deliberately over-noise along a
  stable direction (can add "snap"/contrast at high CFG).

### `noise_paths`
- `1` → plain ancestral noise (merging is a no-op).
- `2` → classic A2.
- `3–5` → sweet spot for noticeably smoother results.
- `6–8` → diminishing returns; costs one extra latent-sized noise draw per path.

### `merge_mode`
- `mean` — variance reduction 1/N; unbiased. Default.
- `median` — per-pixel median; robust to outlier draws but needs **3+ paths**
  to be meaningful (torch takes the lower middle for even N, so with N = 2 it
  just picks one draw per pixel).

### `noise_normalization`
- `none` — historical A2 behavior; merged noise keeps its reduced variance and
  `extrapolation` compensates.
- `variance` — multiplies by √N so the merged noise is unit-variance in
  expectation. Set `extrapolation = 0` for energy preservation.
- `rms` — normalizes each sample's actual RMS to 1 every step. Strongest
  stabilization: identical noise energy for every step and every seed. Also
  useful with `noise_paths = 1` to normalize single draws.

### `active_start` / `active_end`
Restricts the enhanced path to a window of the trajectory, measured as a
fraction of steps (0 = first step, 1 = last). Example: `0.0 – 0.6` stabilizes
the structure-forming early steps, then returns to plain ancestral noise for
fine detail in the late steps.

### `method`
Integration order of the deterministic down-step. `euler` is the classic
behavior. `heun` (2nd order) is the best accuracy-per-eval upgrade and
additionally reuses its endpoint evaluation when substeps chain. `midpoint`
(2nd order) often behaves similarly; `rk4` (4th order) is the most accurate
and the most expensive. Most visible at low step counts or with `eta = 0`.

### `substeps` / `substep_mode` / `substep_spacing`
Split each interval into finer internal segments. Use `ancestral` mode for a
denser noise trajectory (equivalent to a finer schedule), `deterministic`
mode to improve integration accuracy while leaving the noise structure
untouched. `log` spacing matches how diffusion schedules are usually built.

---

## Recommended Recipes

| Goal | Settings |
|---|---|
| **Classic A2** (original behavior) | all defaults |
| **Smooth & stable** | `noise_paths = 4`, `noise_normalization = variance`, `extrapolation = 0` |
| **Maximum stability** | `noise_paths = 5`, `merge_mode = median`, `noise_normalization = rms`, `extrapolation = 0` |
| **Stable structure, lively detail** | `noise_paths = 4`, `noise_normalization = variance`, `extrapolation = 0`, `active_start = 0.0`, `active_end = 0.6` |
| **Normalized single noise** | `noise_paths = 1`, `noise_normalization = rms`, `extrapolation = 0` |
| **Deterministic** | `eta = 0` (noise settings are ignored) |
| **High-accuracy deterministic** | `eta = 0`, `method = rk4`, `substeps = 2`, `substep_mode = deterministic` |
| **Accurate ancestral, same noise feel** | `method = heun`, `substeps = 2`, `substep_mode = deterministic` |
| **Dense ancestral trajectory** | `substeps = 2–4`, `substep_mode = ancestral`, `substep_spacing = log` |

---

## Behavior & Compatibility Notes

- **Backward compatible:** with default widget values the sampler reproduces
  the original Euler A2 output (same RNG consumption order; differences are at
  float32 rounding level, ~4e-8). Old workflows load unchanged — the original
  widgets keep their order and all newer inputs (including `method`,
  `substeps`, `substep_mode`, `substep_spacing`) are appended after them with
  defaults.
- **Deterministic:** same seed + same settings → identical output. Noise is
  drawn from ComfyUI's seeded `default_noise_sampler`, N draws per stochastic
  (sub)step, in order.
- **Ancestral substep identity:** `substep_mode = ancestral` with N substeps
  produces the same trajectory as running on the correspondingly subdivided
  sigma schedule (verified to ~1e-7).
- **Parametrization:** the renoise math assumes `alpha = 1 - sigma` (the
  schedule family the sampler was designed for). With schedules where this
  does not hold, results still run but the noise budget is approximate.
- **Degenerate settings:** `eta = 0` or `s_noise = 0` short-circuit to the
  deterministic path; `noise_paths = 1` with `noise_normalization = none` and
  `extrapolation = 0` is exactly plain Euler ancestral.
- **Edge cases handled:** zero sigmas at either end of a step, zero noise
  scale (skips RNG draws), out-of-range `sigma_down` from extreme `eta`, and
  ODE segments ending at `sigma = 0` (Euler fallback).

---

## Technical Details

### Sampling function signature

```python
sample_euler_a2(
    model, x, sigmas,
    extra_args=None, callback=None, disable=None, noise_sampler=None,
    eta=1.0, s_noise=1.0, extrapolation=0.425,
    noise_paths=2, merge_mode="mean", noise_normalization="none",
    active_start=0.0, active_end=1.0,
    method="euler", substeps=1, substep_mode="ancestral", substep_spacing="log",
)
```

Compatible with ComfyUI's k-diffusion sampler calling convention; the seed is
read from `extra_args["seed"]` when no explicit `noise_sampler` is supplied.

### NFE (model evaluations) per sampling step

| Configuration | Evals per step |
|---|---|
| `euler`, `substeps = 1` | 1 |
| `midpoint` / `heun`, `substeps = 1` | 2 |
| `rk4`, `substeps = 1` | 4 |
| `euler`, `substeps = M` (deterministic mode) | M |
| `heun`, `substeps = M` (either mode) | M + 1 (endpoint reuse) |
| `midpoint`, `substeps = M` | 2M |
| `rk4`, `substeps = M` | 4M − 1 (endpoint reuse) |
| `euler`, `substeps = M` (ancestral mode) | M |

### Callback payload

Per step, the callback receives:

```python
{"x": x, "i": i, "sigma": sigma_i, "sigma_hat": sigma_down, "denoised": denoised}
```

`sigma_hat` reports the actual intermediate down-step sigma (more accurate for
previews and hooks than repeating `sigma_i`). On degenerate steps it equals
`sigma_i`. The callback fires once per outer step, not per substep.

### Registration

On module import, `_register_sampler()` (idempotent):

- sets `comfy.k_diffusion.sampling.sample_euler_a2`
- appends `euler_a2` to `comfy.samplers.KSAMPLER_NAMES`,
  `comfy.samplers.SAMPLER_NAMES`, and `comfy.samplers.KSampler.SAMPLERS`
  (skipping duplicates and missing attributes)

### Node plumbing

- `NODE_CLASS_MAPPINGS = {"Euler_A2_Sampler": EulerA2Sampler}` — the ID must
  stay unchanged for saved workflows to resolve.
- `get_sampler(...)` forwards every widget into
  `comfy.samplers.ksampler("euler_a2", extra_options)`.
- All widgets carry tooltips (shown by newer ComfyUI frontends), and the node
  exposes a `DESCRIPTION` string.

---

## FAQ

**How is this different from `euler_ancestral`?**
`euler_ancestral` injects one fresh noise draw per step and integrates
first-order. Euler A2 merges N draws into a stabilized direction, controls its
gain/variance and active range, and can integrate the deterministic part at
up to 4th order with internal substeps.

**Do higher-order methods help when `eta = 1`?**
Less than with `eta = 0`: in full ancestral sampling the injected noise
dominates the trajectory. They still reduce the systematic bias of the
down-step, most visibly at low step counts. Pair them with
`substep_mode = deterministic` so the noise structure stays unchanged.

**What's the difference between `substeps` and just using more steps?**
`substep_mode = ancestral` is equivalent to a finer schedule (verified), just
configured from inside the sampler. `substep_mode = deterministic` has no
schedule equivalent: it refines only the ODE down-step and keeps the number
and size of noise injections fixed.

**Why does the median mode want 3+ paths?**
A per-pixel median of 2 values just picks one of them (torch takes the lower
middle), so no stabilization happens. Use an odd count ≥ 3.

**What `extrapolation` should I use with normalization enabled?**
`0`. Normalization already restores unit noise energy, so any positive
extrapolation is a deliberate over-noise on top of that.

**Do substeps or higher-order methods slow down generation?**
Yes — both multiply model evaluations per step (see the NFE table). Noise-path
merging is nearly free; the order/substep options trade compute for accuracy.

**Can I use it outside the node, e.g. as a sampler name?**
Yes — after import, `euler_a2` is registered like a built-in sampler and can
be selected anywhere sampler names are accepted; extra options fall back to
the function defaults.
