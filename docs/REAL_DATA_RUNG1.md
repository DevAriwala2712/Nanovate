# Track C Rung 1: frozen v3 verifier on real SEM images of defect-free hole arrays

**REAL DATA.** The reference is a **FITTED lattice, not CAD**.
- Pre-registration: `docs/REAL_DATA_PREREG.md`, commit f734a43.
- The run: `python -m experiments.real_rung1 --run`, at commit ead661f (one crash fix, see §5), on 2026-10-19, 54 s wall time.
- Outputs: `results/real/rung1/` (`tiles.csv`, `summary.csv`, `calibration.csv`, `hole_residuals.csv`, `manifest.csv`, `config.json`, example PNGs).
- Post-hoc diagnostics: `scripts/real/diagnose_rung1.py` writes `diag_*.png` and `diagnostic.csv`.

## 1. What was tested

The frozen v3 verifier (`results/v3/calib/decisions.json`, unchanged) was run on real top-view SEM images of FIB-milled hole arrays in a Pd film:
- **Source:** Yang et al., ACS Nano 2023; doi:10.5281/zenodo.8060099, CC-BY-4.0.
- **Instrument:** Thermo Helios G4 CX, TLD/SE, 10–15 kV.
- **Pixel size:** 5.1–20.8 nm/px, from the FEI metadata.

By visual check, every image has every designed hole, none merged and no extra holes. Any KILLER verdict on these tiles is therefore a false alarm.

| Population | Tiles | Images |
|---|---|---|
| Arm C (native px, primary): G29 (array v2, 5.1–7.7 nm/px), overlay-free tiles | 116 | 29 |
| Arm C: G01 (array v1, 10.8–20.8 nm/px) | 42 | 3 |
| Arm B (diameter-matched, secondary): G01 only | 46 | 3 |
| Overlay tiles (burned-in display cross), reported apart | 58 | 29 |
| Spot-check: InGaN figure panels (not clean: pits between holes) | 6 | 2 |

**Module 1 stand-in scribe patches:**
- **G29:** tile (0, 0) of `…120_dset_200_060`, `…240_dset_194_075` and `…280_dset_260_047`.
- **G01:** tiles (0, 0), (0, 2), (0, 4) of `z_110_overview_002`.

**Reference (FITTED):**
- Lattice basis fitted on the calibration images or region only.
- Hole diameter: the design value from the file name (G29) or fitted on the calibration region (G01, 234 nm).
- Translation registered per image.

## 2. Results vs pre-registered targets

95% CIs come from a bootstrap over images (tiles of one image resampled together, 1000 resamples, seed 0).

| Metric | Population | Result | Target | Met? |
|---|---|---|---|---|
| **KILLER on clean tiles (false alarm)** | **Arm C, G29 + G01 (primary)** | **78.5% [66.4, 87.6]** (n = 158 tiles, 32 images) | ≤ 5% | **NO** |
| KILLER on clean tiles | Arm C, G29 | 70.7% [60.3, 81.9] (116 tiles, 29 images) | — | — |
| KILLER on clean tiles | Arm C, G01 | 100% [100, 100] (42 tiles, 3 images) | — | — |
| KILLER on clean tiles (secondary) | Arm B, G01 | 100% [100, 100] (46 tiles, 3 images) | ≤ 5% | **NO** |
| NEAR_MISS on clean tiles | Arm C, G29 + G01 | 18.4% [9.9, 29.0] (158, 32) | none | — |
| Called NUISANCE (correct) | Arm C, G29 + G01 | 5 of 158 tiles (3%) | — | — |
| KILLER on overlay tiles (not in target) | Arm C, G29 | 65.5% [50.0, 81.0] (58, 29) | — | — |
| KILLER, spot-check (not clean, descriptive) | InGaN 2(b) / 2(d) | 2 of 3 / 0 of 3 tiles (2(d): 3 NEAR_MISS) | — | — |

**Calibration error** (no target):

| Measure | G29 | G01 |
|---|---|---|
| ‖T − I‖/‖I‖ | 0.118 [0.114, 0.123] (174 tiles) | 0.102 [0.097, 0.107] (42) |
| ‖T − I‖/‖I − mean‖ (contrast-relative) | 0.86 [0.83, 0.89] | 0.92 [0.88, 0.96] |
| Contrast-relative, noise-corrected | 0.83 | 0.86 |
| In-sample, on the calibration tiles themselves (contrast-relative) | 0.73–0.83 | 0.78–0.96 |
| Module 1 film-minus-hole contrast | +70.1 grey levels | **−7.8 (sign inverted)** |

The synthetic transfer error ‖T − E‖/‖E‖ was 0.11–0.12 (`results/m3_calibration/`). That synthetic number uses the noise-free mean image E, and the ratio is dominated by the mean level. So "0.12 real vs 0.12 synthetic" is **not** evidence of good transfer. The contrast-relative error shows the template explains only about 15–25% of the real image's variation, and that holds even on the very patches it was fitted on.

**Alignment error** (no target):

| Measure | G29 | G01 |
|---|---|---|
| Hole centre to fitted-reference centre | 6.9 nm [6.1, 7.7] (171 holes, 29 images), about 1.3 px | 23.2 nm [21.2, 25.4] (255 holes, 2 images), about 2 px |
| Module 3 residual shift per tile | 1.65 px [1.45, 1.85] (174 tiles) | 20.0 px [8.7, 33.5] (42 tiles), max 138 px: Module 3 sometimes locks onto a wrong lattice period |

**Run time** (this Mac CPU): calibration 46 ms (3 patches). Per tile 178 ms [177, 179] (G29) and 180 ms (G01), against 130 ms per patch in synthetic Exp 6.

## 3. Where it went wrong (examples with the verdict reason)

The pre-registered rule takes the first five clean, overlay-free arm C KILLER tiles, sorted by group, image, tile. All five fall in G01 `center_overview_001`, row 0:

| Tile | Verdict | Δβ (β₀, β₁) | Bridge / open events | Bottleneck | Max SDF energy |
|---|---|---|---|---|---|
| (0, 0) | KILLER_BRIDGE | (24, 20) | 20 / 17 | 0.383 | 27.1 |
| (0, 1) | KILLER_BRIDGE | (26, 41) | 44 / 15 | 0.430 | 103.6 |
| (0, 2) | KILLER_BRIDGE | (33, 18) | 18 / 18 | 0.350 | 25.8 |
| (0, 3) | KILLER_BRIDGE | (13, 88) | 88 / 9 | 0.431 | 24.5 |
| (0, 4) | KILLER_BRIDGE | (16, 14) | 14 / 10 | 0.276 | 32.0 (Module 3 shift 74 px) |

Post-hoc diagnostics (first clean tile per group and verdict; `diag_*.png`):
- **G29 `…120_dset_226_061` (0, 0):** KILLER_BRIDGE, Δβ (1, 1), one new-hole event and one particle event, bottleneck 0.239, SDF energy 11.7. A sliver of field > 0.5 forms inside the hole rim.
- **G29 `…160_dset_224_077` (0, 0):** KILLER_OPEN/PARTICLE, Δβ (1, 0), bottleneck 0.449, SDF energy 12.2. A ring of field > 0.5 forms inside the real hole: a fake conductor island. The real hole is about 41 nm wider than its 224 nm design diameter.
- **G29 `…160_dset_180_057` (0, 2):** NEAR_MISS, Δβ (0, 0), bottleneck 0.124, SDF energy 5.0 (above e* 2.57).

## 4. What broke (post-hoc diagnosis; nothing was changed)

1. **Module 1 calibration fails on G01 (the main cause of the 100%).** The five-feature linear template explains the bright secondary-electron edge halo around each small hole with the gradient features and gets a **negative** film-minus-hole contrast (−7.8 grey levels). The template-anchored field divides the residual by that contrast. Noise is amplified about 10× and inverted, and on the diagnosed tile 75% of film pixels fall below the 1 − τ level. The result is dozens of fake components and holes per tile. G29's larger holes give a sane contrast (+70).
2. **Real hole edges don't look like the testbed's.** The real holes are black inside with a wide bright halo; the testbed bloom model is narrower. The linear template can't reproduce this, so at the rim the residual pushes the field above 0.5 in a ring or sliver. That reads as a particle (Δβ₀ = +1) or a ring-enclosed extra hole (Δβ₁ = +1).
3. **Reference diameter mismatch.** The real G29 holes are 0–59 nm wider than the design value used as reference (median about +40 nm). Across the 29 images, the mismatch correlates with the KILLER rate (Spearman ρ = 0.43) and more strongly with KILLER_OPEN/PARTICLE (ρ = 0.52). It is not the whole story: the image with a −18 nm mismatch was 4 of 4 KILLER_BRIDGE.
4. **Module 3 on periodic patterns.** With a 135 px pitch and only a few holes per tile, phase correlation sometimes jumps by up to 138 px (G01). That misplaces the reference by a whole lattice cell or more.
5. **Arm B (diameter-matching) did not help.** It was still 100% on G01. Resampling doesn't fix a calibration whose contrast sign is wrong.

## 5. Disclosures

- **Crash and rerun.** The first `--run` crashed on its first tile with a `NameError` (a missing import). Nothing was written or seen. The fix (commit ead661f, plus an end-to-end smoke test) changed no setting, and the run was repeated once. This is allowed by §3 of the pre-registration.
- **Clean labels.** "Clean" is a visual judgement by the AI assistant, made before the run, not an expert label. Sub-resolution defects can't be excluded. But a 78% KILLER rate is far above any plausible true-defect rate on these arrays: the verdicts fire on almost every hole.
- **Example selection.** The five pre-registered examples all come from one G01 image. The G29 diagnostics were chosen after the run by a rule written in the script (first tile per group and verdict) and are labelled post-hoc.

## 6. What this does and does NOT tell us

**Does tell us:**
- On these real images, the frozen verifier is not usable as-is: about 4 in 5 clean tiles get a KILLER call.
- The first real-data bottleneck is **Module 1 calibration transfer** (the linear template plus the contrast estimate), with the edge-halo appearance and alignment on periodic patterns close behind. The topology rule itself never sees a sensible field here.
- Measuring the calibration error contrast-relative matters. The synthetic-style metric looked fine (0.12) and hid this.

**Does NOT tell us:**
- **Recall:** no defects were present.
- How the verifier would do with **true CAD**: the reference here is fitted, with a known diameter mismatch.
- Behaviour on **lines, gaps or bridges** at the **14 nm scale** it was designed for. These are 175–300 nm FIB holes in metal, the inverse polarity of testbed family C.
- Whether a better calibration would fix it. That needs a change chosen on **synthetic validation data** (the rule), then one repeat of this run.

## 7. Candidate next steps (not done; any change chosen on synthetic val, never on these images)

- **Sanity checks on the Module 1 contrast:** refuse to verify, or flag "calibration failed", when the predicted contrast sign or size is implausible. This is a guard, not a tuned threshold.
- **A richer edge model in the simulator:** a wider SE halo in the testbed. Then re-check on synthetic val whether the template and field survive it.
- **Module 3 alignment on periodic patterns:** restrict the shift to less than half a pitch, or align against the whole image instead of each tile.
- **Reference diameter:** for Rung 2, true CAD plus a separate measured-CD map would remove the mismatch confound.
