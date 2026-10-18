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
        Δβ=0, bottleneck ≥ d*, SDF energy ≥ e* (v2: and peak-z ≥ z*) -> NEAR_MISS   otherwise -> NUISANCE
        (every verdict returns its reason: Δβ, bottleneck distances, max SDF energy, peak-z)
```

Baselines on identical patches and labels: **die-to-die** (single and double difference; aligned with the same
Module-3 aligner and stabilised with the same Module-1 noise model; rules fitted per dose on Manhattan val) and a
**small ResNet CNN** on `[image, CAD]` trained on Manhattan only (weights `m6_cnn/cnn.pt`).

## How to run

```bash
python -m venv .venv && .venv/bin/pip install -e .      # numpy scipy scikit-image gdstk gudhi torch matplotlib pandas pytest
.venv/bin/pytest -q                                      # all tests
.venv/bin/python scripts/reproduce.py --quick            # every experiment end to end, small n, a few minutes -> results/v2/quick/
.venv/bin/python scripts/reproduce.py                    # full v2 run -> results/v2/ and the v2 table below (CPU, about an hour)
.venv/bin/python scripts/reproduce.py --version v1       # the first pre-registered run -> results/ and the v1 table
```

**v1 and v2.** v1 is the first, pre-registered run and is kept unchanged. v2 adds fixes designed *after* the v1 test
results were seen (splice-seam holes, an LER-robust near-miss rule, faster calibration); every v2 choice was made on
Manhattan val only and no target was changed. What changed, why, and the val evidence: [`docs/V2_FIXES.md`](docs/V2_FIXES.md).

`reproduce.py` selects the verifier configuration on **Manhattan validation only** (`results/[v2/]calib/decisions.json`),
runs Experiments 1–6 on the held-out **test** split, and rewrites the results block below **from the CSVs only**. One seed
drives everything: non-timing CSVs are identical across runs on the same machine (timings in Exp 6 are not).
The CNN is trained off-machine (`scripts/train_cnn.py`, `notebooks/train_cnn_colab.ipynb`) and only evaluated here.
Per-milestone measurements, surprises and missed targets are in `PROGRESS.md`; the design in `PLAN.md`.

**Demo.** `notebooks/demo.ipynb` walks one synthetic tile through every stage (calibration, alignment, field, persistence diagrams,
SDF energy) and prints each verdict with its reason, for a bridge, an open, a particle, a nuisance sample and a near-miss, plus the
singleton-vs-stack distinction. It is a thin caller of `src/nanovate/demo.py` (tested; `tests/test_demo.py` executes every cell without
jupyter). Regenerate the notebook with `python scripts/make_demo_notebook.py`. The tiles are illustrations from the validation split, not results.

## Why use this: the day-0 case

Existing review-SEM classifiers are trained per defect class and per layer, so on the first wafers of a new layer
they do not exist yet. This verifier needs no labelled images: the table below compares the options available
on that day, generated from the result CSVs. The full argument is in [`docs/USE_CASE.md`](docs/USE_CASE.md).

<!-- DAY0:START -->
_All numbers are **synthetic data**; the review queue is a **projection** from measured per-kind rates (via family, dose 50) with an assumed site mix._

#### Day 0 on a new layer: what each option gives you

|  | Verifier (ours) | CNN (Manhattan-trained) | Die-to-die (single) |
|---|---|---|---|
| Labelled images needed before first use | none (3 scribe images per layer) | a labelled training set (Manhattan here) | labelled val per dose, to fit the rule |
| F1 on a new layout: via | 0.88 [0.82, 0.93] (n=144) | 0.74 [0.66, 0.81] (n=144) | 0.56 [0.45, 0.65] (n=144) |
| F1 on a new layout: curvy | 0.84 [0.77, 0.90] (n=144) | 0.85 [0.79, 0.90] (n=144) | 0.55 [0.45, 0.66] (n=144) |
| F1 on the layout the CNN was trained on (Manhattan) | 0.81 [0.73, 0.87] (n=144) | 0.94 [0.89, 0.97] (n=144) | 0.57 [0.45, 0.67] (n=144) |
| Setup per layer | 43 ms calibration | training run (GPU) | rule fit |
| Time per patch (this CPU) | 99 ms | 3 ms | 31 ms |
| Says why it flagged a site | yes: delta-beta, bottleneck, SDF energy | no (a probability) | no (a difference map) |

Where the verifier loses: on a layout the CNN was trained on, the CNN is more accurate, and it is much faster per patch.

#### Review queue: 1,000 flagged sites on a new via layer, 5% real killers (50 sites)

| Sites (95% CI) | Verifier (no labels) | CNN (Manhattan-trained) | Die-to-die (single) | Die-to-die (double) |
|---|---|---|---|---|
| Killers caught (flagged KILLER) | 45 [42, 48] | 38 [34, 41] | 26 [22, 29] | 31 [26, 35] |
| Killers flagged NEAR_MISS (still reviewed) | 3 [1, 6] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| Killers missed (passed as harmless) | 2 [0, 4] | 12 [9, 16] | 24 [21, 28] | 19 [15, 24] |
| Non-killer sites flagged KILLER (incl. near-misses) | 22 [12, 34] | 73 [28, 146] | 46 [40, 51] | 103 [61, 176] |
| Near-misses flagged NEAR_MISS (reliability bin) | 73 [61, 83] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| Harmless sites flagged NEAR_MISS | 36 [0, 107] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| All sites sent to an engineer | 179 [142, 250] | 111 [66, 185] | 71 [63, 78] | 133 [91, 209] |

Site mix assumed: killers split equally bridge / open / particle; harmless sites 90% roughness-only, 5% notch, 5% filament (near-misses). Only the verifier has a NEAR_MISS bin; the others must call a near-miss killer or harmless.

Other prevalences (killers missed / non-killer sites flagged KILLER):

| Real killers | Verifier (no labels) | CNN (Manhattan-trained) | Die-to-die (single) | Die-to-die (double) |
|---|---|---|---|---|
| 1% | 0 [0, 1] / 23 [12, 35] | 2 [2, 3] / 76 [29, 153] | 5 [4, 6] / 47 [41, 54] | 4 [3, 5] / 107 [64, 184] |
| 20% | 8 [0, 17] / 18 [10, 28] | 50 [36, 64] / 62 [23, 123] | 97 [83, 111] / 38 [33, 43] | 78 [61, 94] / 87 [52, 148] |

#### Label break-even

The label break-even experiment (how many labelled images of a new layout the CNN needs to match the verifier) is **not yet run**.
<!-- DAY0:END -->

## Results (v2: post-hoc fixes)

<!-- RESULTS:START -->
_All numbers below are **synthetic data** (full run, v2; 95% percentile-bootstrap CIs in brackets, n beside every number)._

_v2 is a **post-hoc revision**: its fixes were designed after the v1 test results were seen (splice-seam holes, LER-robust near-miss selection, faster calibration; see `docs/V2_FIXES.md`). Every v2 choice was made on Manhattan val only and the targets are unchanged; v1 numbers are kept below._

_Verifier config frozen on Manhattan val: field `template`, region `splice`, tau 0.35, pre-smooth 1.0 px, d* 0.0775434990066642, e* 2.6189942373880593, z* 5.578071347336651, holes `dielectric` (val killer F1 0.86, n=288)._

#### Pre-registered targets vs measured (verifier)

| Exp | Target (hypothesis) | Measured | Met |
|---|---|---|---|
| 1 Nuisance stress | >= 90% suppression and >= 90% bridge recall at LER sigma <= 1.5 nm | worst cell: suppression 88%, bridge recall 79% | **NO** |
| 2 Dose cliff | F1 >= 0.9 at lambda >= 50 e-/px | F1 at lambda >= 50: min 0.81 (max 0.85) | **NO** |
| 3 Zero-shot geometry | F1 drop A -> B/C <= 5 points | curvy: -3.7 pts; via: -7.3 pts | yes |
| 4 Misalignment | >= 95% verdicts unchanged (shift +-8 px + warp) | 94% with Module 3 | **NO** |
| 5 Near-miss capture | recall >= 80% at the fixed false-alarm rate | 90% recall | yes |
| 6 Calibration time | < 50 ms for 3 scribe patches | 43 ms (median of 100) | yes |

#### Exp 1 - nuisance stress (LER sigma 1.5 nm, grain 30%; full grid in `results/v2/exp1/`)

| Method | Nuisance suppression (NUISANCE verdict) | Nuisance not flagged KILLER (secondary) | Bridge recall |
|---|---|---|---|
| Verifier (ours) | 88% [75, 100] (n=24) | 92% [79, 100] (n=24) | 79% [62, 96] (n=24) |
| Die-to-die (single) | 29% [12, 50] (n=24) | 29% [12, 50] (n=24) | 92% [79, 100] (n=24) |
| Die-to-die (double) | 21% [4, 38] (n=24) | 21% [4, 38] (n=24) | 100% [100, 100] (n=24) |
| CNN (Manhattan-trained) | 92% [79, 100] (n=24) | 92% [79, 100] (n=24) | 92% [79, 100] (n=24) |

#### Exp 2 - killer F1 vs dose (e-/px)

| Method | 5 | 10 | 20 | 35 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|---|
| Verifier (ours) | 0.67 [0.60, 0.74] (n=144) | 0.68 [0.60, 0.76] (n=144) | 0.81 [0.73, 0.87] (n=144) | 0.84 [0.78, 0.90] (n=144) | 0.81 [0.73, 0.87] (n=144) | 0.84 [0.77, 0.89] (n=144) | 0.85 [0.78, 0.90] (n=144) |
| Die-to-die (single) | 0.62 [0.53, 0.71] (n=144) | 0.69 [0.61, 0.76] (n=144) | 0.62 [0.52, 0.70] (n=144) | 0.61 [0.50, 0.71] (n=144) | 0.57 [0.45, 0.67] (n=144) | 0.66 [0.56, 0.74] (n=144) | 0.62 [0.52, 0.72] (n=144) |
| Die-to-die (double) | 0.68 [0.60, 0.75] (n=144) | 0.70 [0.61, 0.78] (n=144) | 0.63 [0.54, 0.71] (n=144) | 0.68 [0.60, 0.75] (n=144) | 0.61 [0.51, 0.71] (n=144) | 0.59 [0.47, 0.69] (n=144) | 0.66 [0.59, 0.73] (n=144) |
| CNN (Manhattan-trained) | 0.72 [0.62, 0.81] (n=144) | 0.86 [0.79, 0.92] (n=144) | 0.93 [0.89, 0.97] (n=144) | 0.90 [0.86, 0.95] (n=144) | 0.94 [0.89, 0.97] (n=144) | 0.95 [0.91, 0.98] (n=144) | 0.95 [0.90, 0.98] (n=144) |

Lowest dose with F1 >= 0.9: Verifier (ours): none; Die-to-die (single): none; Die-to-die (double): none; CNN (Manhattan-trained): 20.0

#### Exp 2 diagnostic - verifier ceiling without shot noise (not a target)

| Input | Killer F1 | Nuisance suppression | Recall bridge / open / particle |
|---|---|---|---|
| verifier (noisy, dose 50) | 0.81 [0.73, 0.87] (n=144) | 96% | 83% / 75% / 100% |
| verifier (noise-free expected image) | 0.84 [0.77, 0.89] (n=144) | 96% | 79% / 88% / 100% |

Killers missed even noise-free (below this verifier's optical resolution): bridge 5/24; open 3/24; particle 0/24.

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
| verifier without M3 | 45% [38, 52] (n=192) | 74% [68, 81] (n=192) |

#### Exp 5 - near-miss capture

| Kind | Rule | Metric | Rate |
|---|---|---|---|
| nuisance | delta beta alone | alarm_rate | 4% [0, 10] (n=72) |
| nuisance | full rule | alarm_rate | 4% [0, 10] (n=72) |
| notch | delta beta alone | recall | 42% [31, 53] (n=72) |
| notch | full rule | recall | 79% [69, 88] (n=72) |
| filament | delta beta alone | recall | 17% [10, 25] (n=72) |
| filament | full rule | recall | 100% [100, 100] (n=72) |
| notch+filament | delta beta alone | recall | 29% [22, 37] (n=144) |
| notch+filament | full rule | recall | 90% [84, 94] (n=144) |

#### Exp 6 - latency (ms, this CPU; not bit-reproducible)

| What | Median | Mean [95% CI] (n) |
|---|---|---|
| calibration (3 scribe patches, with registration) | 43.1 | 43.1 [43.1, 43.2] (n=100) |
| calibration (3 scribe patches, no registration) | 15.7 | 15.7 [15.7, 15.7] (n=100) |
| verifier: Module 3 alignment | 16.2 | 16.2 [16.2, 16.2] (n=1000) |
| verifier: Module 3 SDF energy | 6.9 | 7.0 [7.0, 7.1] (n=1000) |
| verifier: Module 4 topology (+ field) | 75.9 | 78.4 [77.9, 78.9] (n=1000) |
| verifier: total per patch (singleton path) | 99.2 | 101.6 [101.1, 102.1] (n=1000) |
| die-to-die single (align + GAT + stats) | 31.2 | 31.2 [31.2, 31.2] (n=1000) |
| cnn batch 1 (default threads) | 2.5 | 2.6 [2.6, 2.6] (n=1000) |
| verifier: stack path total per patch (K=32) | 109.3 | 109.3 [109.3, 109.3] (n=1) |
| cnn batch 1, 1 thread(s) | 3.2 | 3.3 [3.2, 3.3] (n=20) |
| cnn batch 64, 1 thread(s) (per patch) | 7.9 | 7.9 [7.8, 8.1] (n=5) |
| cnn batch 1, 5 thread(s) | 2.2 | 2.3 [2.2, 2.4] (n=20) |
| cnn batch 64, 5 thread(s) (per patch) | 7.6 | 7.6 [7.5, 7.6] (n=5) |
<!-- RESULTS:END -->

## Results (v1: first pre-registered run, kept unchanged)

<!-- RESULTS_V1:START -->
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
<!-- RESULTS_V1:END -->

## Limitations (read these before believing any number above)

- **Sim-to-real gap.** All data is synthetic, from a forward model written by the same authors. The testbed adds deliberate
  mismatch (anisotropic beam, saturating bloom, grain, charging) so Module 1 is not circular, but real SEM physics
  (charging dynamics, material contrast, sample damage) is not modelled. Ranking methods on this testbed says little about real tools.
- **τ sensitivity.** A dim bridge of conductor level *c* is detected only if *c* > 1 − τ, so τ trades dim-bridge recall against
  bloom-induced fake bridges. τ, the field option and the near-miss thresholds were selected on Manhattan validation (`results/calib/`), and the choice is
  sensitive: see `results/calib/candidates.csv`. Grain makes it worse: at 30% grain contrast a dark grain on a narrow bridge
  drops it below the merge level (Exp 1 bridge recall 79%); a larger τ recovers those bridges but loses opens (`docs/V2_FIXES.md`).
- **Optical resolution.** Cuts shorter than ~4–5 px and particles within ~3–4 px of a wire are barely visible in the simulated images,
  so some ground-truth killers are sub-resolution; most 1–3 nm filaments read as bridges. These are reported as misses, not tuned away.
- **Singleton layouts.** Module 2 needs same-layout peers (K = 32). Without them the pipeline falls back to Gaussian smoothing and flags
  the result `singleton=True`; in these measurements the stack field did not beat per-patch smoothing for Module 4 (better particles, worse opens).
- **Scribe-to-die transfer.** Calibration comes from three large isolated features; measured transfer error on in-die patches is 11–14%
  (`PROGRESS.md` M3), and alignment against the CAD-derived template has a floor of ~0.2–0.9 px from LER.
- **Patch boundaries.** Wires continue outside the crop; topology is compared on a margin-aware window and the verifier's region tolerance
  (`region_tol` = 8 px) must absorb alignment error, which creates seam artifacts. v2 removes the seam's fake holes
  (holes counted with dielectric outside the region); a real bridge whose new hole touches the seam is then missed.
- **v2 is a second look at test.** Its fixes were designed after the v1 test results were seen; all v2 choices were made on
  Manhattan val, but the near-miss selection uses LER up to 1.5 nm because Exp 1 tests that range. v1 is kept for comparison.
- **Baselines.** Die-to-die rules are tuned per dose with *labelled* validation data (generous to the baseline); the CNN sees the CAD channel
  and was trained off-machine on a GPU (not bit-reproducible).

## Review UI
`frontend/index.html` views testbed samples with their verdicts, ground-truth labels, Module 4 reasons (Δβ, bottleneck, SDF energy) and patch layers (CAD, W, SEM, conductor field).
```
python scripts/export_ui.py                      # writes frontend/data/run.js from the current pipeline (synthetic data)
python3 -m http.server --directory frontend      # then open http://localhost:8000
```
The export runs each sample through the real pipeline with the verifier configuration frozen on Manhattan val (`results/calib/decisions.json`), at one dose (`--dose`, default 50 e-/px).
