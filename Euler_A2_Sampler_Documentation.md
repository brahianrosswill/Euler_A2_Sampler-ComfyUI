# Euler A2 Sampler — Documentation

A custom ancestral sampler node for **ComfyUI**. It extends the classic Euler
ancestral sampler by drawing **multiple independent noise paths per step**,
merging them into a single stabilized noise direction, and extrapolating along
that direction. The result is ancestral sampling with controllable smoothness,
stability, and noise energy.

- Node ID: `Euler_A2_Sampler` (display name: **Euler A2 Sampler**)
- Category: `sampling/custom_sampling/samplers`
- Sampler name registered in ComfyUI: `euler_a2`
- Output: `SAMPLER` (connect to any node that accepts a sampler, e.g. **SamplerCustom**)

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Installation](#installation)
3. [Node Reference](#node-reference)
4. [Parameter Guide](#parameter-guide)
5. [Recommended Recipes](#recommended-recipes)
6. [Behavior & Compatibility Notes](#behavior--compatibility-notes)
7. [Technical Details](#technical-details)
8. [FAQ](#faq)

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

   deterministic_path = (sigma_down / sigma_i) * x
                      + (1 - sigma_down / sigma_i) * denoised
   ```

3. **Renoise base** — the path is rescaled to the target alpha level, and the
   exact noise magnitude needed to reach `sigma_{i+1}` is derived:

   ```
   alpha_{i+1} = 1 - sigma_{i+1}
   alpha_down  = max(1 - sigma_down, eps)
   base        = (alpha_{i+1} / alpha_down) * deterministic_path
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

### Outputs

| Output | Type | Description |
|---|---|---|
| `SAMPLER` | SAMPLER | Configured `euler_a2` sampler instance, ready for SamplerCustom or any SAMPLER input. |

---

## Parameter Guide

### `eta`
Controls how stochastic each step is. `1.0` is standard ancestral behavior.
Lower values shrink the renoise amount; `0` disables noise injection entirely
(pure deterministic Euler/DDIM-like path).

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

---

## Behavior & Compatibility Notes

- **Backward compatible:** with default widget values the sampler reproduces
  the original Euler A2 output (same RNG consumption order; differences are at
  float32 rounding level). Old workflows load unchanged — the first three
  widgets keep their order and all new inputs have defaults.
- **Deterministic:** same seed + same settings → identical output. Noise is
  drawn from ComfyUI's seeded `default_noise_sampler`, N draws per stochastic
  step, in order.
- **Parametrization:** the renoise math assumes `alpha = 1 - sigma` (the
  schedule family the sampler was designed for). With schedules where this
  does not hold, results still run but the noise budget is approximate.
- **Degenerate settings:** `eta = 0` or `s_noise = 0` short-circuit to the
  deterministic path; `noise_paths = 1` with `noise_normalization = none` and
  `extrapolation = 0` is exactly plain Euler ancestral.
- **Edge cases handled:** zero sigmas at either end of a step, zero noise
  scale (skips RNG draws), and out-of-range `sigma_down` from extreme `eta`.

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
)
```

Compatible with ComfyUI's k-diffusion sampler calling convention; the seed is
read from `extra_args["seed"]` when no explicit `noise_sampler` is supplied.

### Callback payload

Per step, the callback receives:

```python
{"x": x, "i": i, "sigma": sigma_i, "sigma_hat": sigma_down, "denoised": denoised}
```

`sigma_hat` reports the actual intermediate down-step sigma (more accurate for
previews and hooks than repeating `sigma_i`). On degenerate steps it equals
`sigma_i`.

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
`euler_ancestral` injects one fresh noise draw per step. Euler A2 merges N
draws into a stabilized direction and lets you control its gain, variance,
and the step range where it applies.

**Why does the median mode want 3+ paths?**
A per-pixel median of 2 values just picks one of them (torch takes the lower
middle), so no stabilization happens. Use an odd count ≥ 3.

**What `extrapolation` should I use with normalization enabled?**
`0`. Normalization already restores unit noise energy, so any positive
extrapolation is a deliberate over-noise on top of that.

**Does more noise paths slow down generation?**
Each path costs one extra latent-sized random tensor per step (cheap) and one
stack/reduce per step. Model inference dominates; the overhead is negligible.

**Can I use it outside the node, e.g. as a sampler name?**
Yes — after import, `euler_a2` is registered like a built-in sampler and can
be selected anywhere sampler names are accepted; extra options fall back to
the function defaults.
