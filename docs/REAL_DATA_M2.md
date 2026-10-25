# Track C real-wiring false-alarm test (M2): frozen verifier on real metal-2 SEM images vs validated labels

**REAL DATA.**
- **Source:** doi:10.17617/3.HY5SYN, CC-BY-4.0; a commercial 128 nm IC, metal 2, 14.65 nm/px, BSE at 15 keV.
- **Pre-registration:** `docs/REAL_DATA_M2_PREREG.md`, commit 2753140, before any verifier run on this data.
- **Run:** once, `python -m experiments.real_m2 --run`. Outputs: `results/real/m2/` (`tiles.csv`, `summary.csv`, `config.json`, `manifest.csv`, example PNGs).

## What was tested

- **Verifier:** the latest frozen state (Rung 1d: v3 rule, guard, template clamp, bounded alignment, near-miss gate d* 0.0798, e* 2.535, z* 5.326), unchanged.
- **Reference:** the authors' manually validated track and via labels (a layout extraction from these images, not designer CAD).
- **Calibration:** the central tile of images 0000, 0160 and 0320. The guard judged it reliable (contrast 71.9, leave-one-out 68.7–74.6).
- **Tiles:** every full 256 x 256 tile of the other 318 images.
- **Label QC:** 1.3% of tiles flagged as touching bright unlabelled structure; reported separately.

## Results vs pre-registered targets

95% CI from a bootstrap over images.

| Metric | Native (primary): 64,536 QC-passed tiles, 318 images | Target | Met? | x0.5 arm: 14,629 tiles |
|---|---|---|---|---|
| **KILLER** | **0.31% [0.24, 0.39]** (197 tiles) | ≤ 5% | **YES** | 1.02% [0.81, 1.28] |
| **Coverage** | **100%** | ≥ 80% | **YES** | 100% |
| **NEAR_MISS** | **76.9% [76.3, 77.4]** | ≤ 5% | **NO** | 84.4% [83.5, 85.4] |

- **QC-flagged tiles** (native, 829): KILLER 1.6%, NEAR_MISS 74%.
- **All tiles** (native, 65,365): KILLER 0.32%, NEAR_MISS 76.8%.
- **Skipped:** 780 tiles where the label raster is all one class.

**Verdicts (native, primary):** 49,611 NEAR_MISS, 14,728 NUISANCE, 184 KILLER_OPEN/PARTICLE, 13 KILLER_BRIDGE. Most killers have Δβ₀ = +1 (143) or +2 (32): an extra conductor component. All five pre-registered KILLER examples (`example_killer_*.png`) contain bright vias.

## Why the near-miss gate fires (post-hoc diagnosis; nothing was changed)

On non-killer native tiles, the gate's three conditions pass at these rates:

| Condition | Real M2 pass rate | Real M2 median | Synthetic val nuisance (Rung 1d held-out) pass rate | Synthetic median |
|---|---|---|---|---|
| bottleneck ≥ 0.080 | 97.5% | 0.126 | 100% | 0.151 |
| **SDF energy ≥ 2.535** | **97.7%** | **13.6** | 3.1% | 1.6 |
| peak-z ≥ 5.33 | 79.5% | 11.3 | 3.1% | 2.8 |

The energy is about 8× the synthetic level and close to its ceiling (residual / contrast x exp(0.2 x 10) ≈ 14.4). It tracks **vias**:

| Vias in the tile | Tiles | NEAR_MISS | Median SDF energy |
|---|---|---|---|
| 0 | 15,567 | 36% | 3.1 |
| 1–2 | 15,997 | 87% | 14.1 |
| 3–5 | 19,422 | 94% | 14.3 |
| 6–10 | 11,458 | 89% | 14.3 |
| > 10 | 2,092 | 75% | 13.9 |

In these BSE images, vias are 2–3× brighter than the tracks they sit on. The verifier's model has one conductor level, so the template predicts track brightness inside every via. That leaves a large, sharp residual deep inside conductor, where the SDF weighting is highest. Even via-free tiles sit just above e* (median 3.1 against 2.535). The testbed has no second bright material, so nothing in synthetic selection could have prepared the gate for this.

## What this tells us

1. **The topology core works on real wiring.** Against a validated layout, at a feature scale close to the testbed, the frozen verifier calls KILLER on 0.31% of clean tiles. That is about 1 in 300 tiles, against Rung 1's 78.5% on FIB hole arrays with a fitted reference. This is the first real-data result that **meets the killer target**, with a large n and a tight CI.
2. **The near-miss gate does not transfer to real images.** It fires on 77% of tiles, driven mainly by a material the model doesn't represent (bright vias). Rungs 1b–1d hinted at the same pattern: the SDF energy is calibrated in synthetic contrast units and saturates on real structure.
3. **Feature-scale matching doesn't help** (x0.5 arm: KILLER 1.0%, NEAR_MISS 84%), consistent with Rung 1's arm B.

## What it does NOT tell us

- **Recall:** the labels come from these same images, so a real defect would be in the reference too.
- **Behaviour against designer CAD**, which would lack the via-brightness and edge-shape detail the labels capture.
- **Whether the 197 KILLER tiles are verifier errors or label errors.** They weren't reviewed one by one; all five illustrated ones involve vias.

## Next (not done)

- **Near-miss gate:** a two-level conductor model (track level, plus a via level from via labels or layout) in Module 1. On real wiring with vias, this is the mechanism-level fix. As always: choose on synthetic data first (add bright vias to the testbed), then one real repeat.
- **Rung 3 (Puschner, 90 nm):** the download is in progress; registration and reference-cell design come next.
