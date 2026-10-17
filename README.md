# Nanovate — training-free SEM topology defect verifier (synthetic-data prototype)

**Thesis.** A SEM defect inspector should call a defect a *killer* only when the circuit's **topology differs from
CAD** (a short, an open, a stray conductive particle), not when pixels merely differ (line-edge roughness, grain,
shot noise, charging, a few nanometres of shift). This prototype tests that idea with no learned model: a per-layer
ridge calibration from three scribe-line patches, sub-pixel alignment to the calibrated template, and a
persistent-homology comparison of the image's conductor field against the CAD (Δβ₀, Δβ₁), with a bottleneck-distance
+ SDF-energy rule for near-misses. It is compared against honest baselines (die-to-die differencing, and a CNN trained
on Manhattan layouts only) on a **synthetic** SEM testbed. **Everything here is measured on simulated data; nothing is
validated on real SEM images** (see Limitations). Targets from the planning documents are hypotheses, reported as
met or missed, never tuned toward.

## Pipeline

```
 3 scribe patches ──► [1] Calibrate: ridge on 5 CAD features -> template T(C), conductor/dielectric levels,
 (per layer)               noise model (GAT)                                    (once per layer)
                                   │
 SEM patch + CAD clip ──► [3] Align to T(C) (sub-pixel, iterative) ──► [2] Stack of K=32 same-layout dies:
                                   │                                       rank-2 SVD + MAD  (singleton: Gaussian
                                   │                                       denoise, flagged singleton=True)
                                   ▼
        field P (conductor probability: template-anchored, calibrated levels)      CAD SDF energy |S|·exp(γ|φ|)
                                   │                                                      │
                                   ▼                                                      │
 [4] Cubical persistence, Betti (lifetime ≥ τ) vs CAD on a margin-aware window ◄──────────┘
        Δβ₀<0 or Δβ₁>0  -> KILLER_BRIDGE        Δβ₀>0 or Δβ₁<0 -> KILLER_OPEN/PARTICLE
        Δβ=0, bottleneck ≥ d*, SDF energy ≥ e* -> NEAR_MISS          otherwise -> NUISANCE
        (every verdict returns its reason: Δβ, bottleneck distances, max SDF energy)
```

Baselines on identical patches and labels: **die-to-die** (single and double difference; aligned with the same
Module-3 aligner and stabilised with the same Module-1 noise model; rules fitted per dose on Manhattan val) and a
**small ResNet CNN** on `[image, CAD]` trained on Manhattan only (weights `m6_cnn/cnn.pt`).

## How to run

```bash
python -m venv .venv && .venv/bin/pip install -e .      # numpy scipy scikit-image gdstk gudhi torch matplotlib pandas pytest
.venv/bin/pytest -q                                      # all tests
.venv/bin/python scripts/reproduce.py --quick            # every experiment end to end, small n, a few minutes -> results/quick/
.venv/bin/python scripts/reproduce.py                    # full run -> results/ and the table below (tens of minutes, CPU)
```

`reproduce.py` selects the verifier configuration on **Manhattan validation only** (`results/calib/decisions.json`),
runs Experiments 1–6 on the held-out **test** split, and rewrites the results block below **from the CSVs only**. One seed
drives everything: non-timing CSVs are identical across runs on the same machine (timings in Exp 6 are not).
The CNN is trained off-machine (`scripts/train_cnn.py`, `notebooks/train_cnn_colab.ipynb`) and only evaluated here.
Per-milestone measurements, surprises and missed targets are in `PROGRESS.md`; the design in `PLAN.md`.

**Demo.** `notebooks/demo.ipynb` walks one synthetic tile through every stage (calibration, alignment, field, persistence diagrams,
SDF energy) and prints each verdict with its reason, for a bridge, an open, a particle, a nuisance sample and a near-miss, plus the
singleton-vs-stack distinction. It is a thin caller of `src/nanovate/demo.py` (tested; `tests/test_demo.py` executes every cell without
jupyter). Regenerate the notebook with `python scripts/make_demo_notebook.py`. The tiles are illustrations from the validation split, not results.

## Results

<!-- RESULTS:START -->
_All numbers below are **synthetic data** (full run; 95% percentile-bootstrap CIs in brackets, n beside every number)._

_Verifier config frozen on Manhattan val: field `template`, region `splice`, tau 0.35, pre-smooth 1.0 px, d* 0.08830792303525908, e* 2.39460919502447 (val killer F1 0.85, n=288)._

#### Pre-registered targets vs measured (verifier)

| Exp | Target (hypothesis) | Measured | Met |
|---|---|---|---|
| 1 Nuisance stress | >= 90% suppression and >= 90% bridge recall at LER sigma <= 1.5 nm | worst cell: suppression 46%, bridge recall 79% | **NO** |
| 2 Dose cliff | F1 >= 0.9 at lambda >= 50 e-/px | F1 at lambda >= 50: min 0.81 (max 0.85) | **NO** |
| 3 Zero-shot geometry | F1 drop A -> B/C <= 5 points | curvy: -3.7 pts; via: -7.3 pts | yes |
| 4 Misalignment | >= 95% verdicts unchanged (shift +-8 px + warp) | 94% with Module 3 | **NO** |
| 5 Near-miss capture | recall >= 80% at the fixed false-alarm rate | 92% recall | yes |
| 6 Calibration time | < 50 ms for 3 scribe patches | 79 ms (median of 100) | **NO** |

#### Exp 1 - nuisance stress (LER sigma 1.5 nm, grain 30%; full grid in `results/exp1/`)

| Method | Nuisance suppression (NUISANCE verdict) | Nuisance not flagged KILLER (secondary) | Bridge recall |
|---|---|---|---|
| Verifier (ours) | 46% [25, 67] (n=24) | 92% [79, 100] (n=24) | 79% [62, 96] (n=24) |
| Die-to-die (single) | 29% [12, 50] (n=24) | 29% [12, 50] (n=24) | 92% [79, 100] (n=24) |
| Die-to-die (double) | 21% [4, 38] (n=24) | 21% [4, 38] (n=24) | 100% [100, 100] (n=24) |
| CNN (Manhattan-trained) | 92% [79, 100] (n=24) | 92% [79, 100] (n=24) | 92% [79, 100] (n=24) |

#### Exp 2 - killer F1 vs dose (e-/px)

| Method | 5 | 10 | 20 | 35 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|---|
| Verifier (ours) | 0.67 [0.60, 0.74] (n=144) | 0.68 [0.59, 0.75] (n=144) | 0.81 [0.73, 0.87] (n=144) | 0.84 [0.78, 0.90] (n=144) | 0.81 [0.73, 0.87] (n=144) | 0.84 [0.77, 0.89] (n=144) | 0.85 [0.78, 0.90] (n=144) |
| Die-to-die (single) | 0.62 [0.53, 0.71] (n=144) | 0.69 [0.61, 0.76] (n=144) | 0.62 [0.52, 0.70] (n=144) | 0.61 [0.50, 0.71] (n=144) | 0.57 [0.45, 0.67] (n=144) | 0.66 [0.56, 0.74] (n=144) | 0.62 [0.52, 0.72] (n=144) |
| Die-to-die (double) | 0.68 [0.60, 0.75] (n=144) | 0.70 [0.61, 0.78] (n=144) | 0.63 [0.54, 0.71] (n=144) | 0.68 [0.60, 0.75] (n=144) | 0.61 [0.51, 0.71] (n=144) | 0.59 [0.47, 0.69] (n=144) | 0.66 [0.59, 0.73] (n=144) |
| CNN (Manhattan-trained) | 0.72 [0.62, 0.81] (n=144) | 0.86 [0.79, 0.92] (n=144) | 0.93 [0.89, 0.97] (n=144) | 0.90 [0.86, 0.95] (n=144) | 0.94 [0.89, 0.97] (n=144) | 0.95 [0.91, 0.98] (n=144) | 0.95 [0.90, 0.98] (n=144) |

Lowest dose with F1 >= 0.9: Verifier (ours): none; Die-to-die (single): none; Die-to-die (double): none; CNN (Manhattan-trained): 20.0

#### Exp 3 - zero-shot geometry (killer F1)

| Method | manhattan | curvy | via |
|---|---|---|---|
| Verifier (ours) | 0.81 [0.73, 0.87] (n=144) | 0.84 [0.77, 0.90] (n=144) | 0.88 [0.82, 0.93] (n=144) |
| Die-to-die (single) | 0.57 [0.45, 0.67] (n=144) | 0.55 [0.45, 0.66] (n=144) | 0.56 [0.45, 0.65] (n=144) |
| Die-to-die (double) | 0.61 [0.51, 0.71] (n=144) | 0.60 [0.50, 0.69] (n=144) | 0.58 [0.48, 0.67] (n=144) |
| CNN (Manhattan-trained) | 0.94 [0.89, 0.97] (n=144) | 0.85 [0.79, 0.90] (n=144) | 0.74 [0.66, 0.81] (n=144) |

F1 drop vs Manhattan (points, 95% CI): Verifier (ours) -> curvy: -3.7 [-12.7, +5.7]; Verifier (ours) -> via: -7.3 [-15.8, +1.6]; Die-to-die (single) -> curvy: +1.2 [-14.0, +15.9]; Die-to-die (single) -> via: +0.5 [-15.4, +14.9]; Die-to-die (double) -> curvy: +1.6 [-12.3, +14.4]; Die-to-die (double) -> via: +3.1 [-9.8, +15.6]; CNN (Manhattan-trained) -> curvy: +9.1 [+1.9, +16.7]; CNN (Manhattan-trained) -> via: +19.9 [+10.8, +29.5]

#### Exp 4 - verdicts unchanged vs zero shift (shift +-8 px + warp <= 1 px)

| Pipeline | Verdict class unchanged | Killer flag unchanged |
|---|---|---|
| verifier with M3 | 94% [91, 97] (n=192) | 95% [92, 98] (n=192) |
| verifier without M3 | 39% [32, 46] (n=192) | 74% [68, 81] (n=192) |

#### Exp 5 - near-miss capture

| Kind | Rule | Metric | Rate |
|---|---|---|---|
| nuisance | delta beta alone | alarm_rate | 4% [0, 10] (n=72) |
| nuisance | full rule | alarm_rate | 4% [0, 10] (n=72) |
| notch | delta beta alone | recall | 42% [31, 53] (n=72) |
| notch | full rule | recall | 83% [75, 92] (n=72) |
| filament | delta beta alone | recall | 17% [10, 25] (n=72) |
| filament | full rule | recall | 100% [100, 100] (n=72) |
| notch+filament | delta beta alone | recall | 29% [22, 37] (n=144) |
| notch+filament | full rule | recall | 92% [87, 96] (n=144) |

#### Exp 6 - latency (ms, this CPU; not bit-reproducible)

| What | Median | Mean [95% CI] (n) |
|---|---|---|
| calibration (3 scribe patches, with registration) | 79.4 | 82.0 [81.1, 83.1] (n=100) |
| calibration (3 scribe patches, no registration) | 27.4 | 27.5 [27.4, 27.7] (n=100) |
| verifier: Module 3 alignment | 16.3 | 16.4 [16.4, 16.5] (n=1000) |
| verifier: Module 3 SDF energy | 6.2 | 6.3 [6.3, 6.3] (n=1000) |
| verifier: Module 4 topology (+ field) | 79.0 | 79.8 [79.5, 80.1] (n=1000) |
| verifier: total per patch (singleton path) | 101.8 | 102.6 [102.2, 102.9] (n=1000) |
| die-to-die single (align + GAT + stats) | 31.4 | 31.6 [31.5, 31.7] (n=1000) |
| cnn batch 1 (default threads) | 2.7 | 2.8 [2.8, 2.8] (n=1000) |
| verifier: stack path total per patch (K=32) | 108.0 | 108.0 [108.0, 108.0] (n=1) |
| cnn batch 1, 1 thread(s) | 3.3 | 3.3 [3.3, 3.3] (n=20) |
| cnn batch 64, 1 thread(s) (per patch) | 8.5 | 8.6 [8.3, 8.9] (n=5) |
| cnn batch 1, 5 thread(s) | 2.5 | 2.7 [2.5, 2.9] (n=20) |
| cnn batch 64, 5 thread(s) (per patch) | 8.2 | 8.2 [8.1, 8.3] (n=5) |
<!-- RESULTS:END -->

## Limitations (read these before believing any number above)

- **Sim-to-real gap.** All data is synthetic, from a forward model written by the same authors. The testbed adds deliberate
  mismatch (anisotropic beam, saturating bloom, grain, charging) so Module 1 is not circular, but real SEM physics
  (charging dynamics, material contrast, sample damage) is not modelled. Ranking methods on this testbed says little about real tools.
- **τ sensitivity.** A dim bridge of conductor level *c* is detected only if *c* > 1 − τ, so τ trades dim-bridge recall against
  bloom-induced fake bridges. τ, the field option and the near-miss thresholds were selected on Manhattan validation (`results/calib/`), and the choice is
  sensitive: see `results/calib/candidates.csv`.
- **Optical resolution.** Cuts shorter than ~4–5 px and particles within ~3–4 px of a wire are barely visible in the simulated images,
  so some ground-truth killers are sub-resolution; most 1–3 nm filaments read as bridges. These are reported as misses, not tuned away.
- **Singleton layouts.** Module 2 needs same-layout peers (K = 32). Without them the pipeline falls back to Gaussian smoothing and flags
  the result `singleton=True`; in these measurements the stack field did not beat per-patch smoothing for Module 4 (better particles, worse opens).
- **Scribe-to-die transfer.** Calibration comes from three large isolated features; measured transfer error on in-die patches is 11–14%
  (`PROGRESS.md` M3), and alignment against the CAD-derived template has a floor of ~0.2–0.9 px from LER.
- **Patch boundaries.** Wires continue outside the crop; topology is compared on a margin-aware window and the verifier's region tolerance
  (`region_tol` = 8 px) must absorb alignment error, which creates seam artifacts.
- **Baselines.** Die-to-die rules are tuned per dose with *labelled* validation data (generous to the baseline); the CNN sees the CAD channel
  and was trained off-machine on a GPU (not bit-reproducible).

## Review UI
`frontend/index.html` views testbed samples with their verdicts, ground-truth labels, Module 4 reasons (Δβ, bottleneck, SDF energy) and patch layers (CAD, W, SEM, conductor field).
```
python scripts/export_ui.py                      # writes frontend/data/run.js from the current pipeline (synthetic data)
python3 -m http.server --directory frontend      # then open http://localhost:8000
```
The export runs each sample through the real pipeline with the verifier configuration frozen on Manhattan val (`results/calib/decisions.json`), at one dose (`--dose`, default 50 e-/px).
