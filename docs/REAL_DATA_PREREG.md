# Track C Rung 1 pre-registration: frozen verifier on real SEM images of defect-free hole arrays

**Written and committed before the verifier is run on any real image.** The run is `python -m experiments.real_rung1 --run`.
It refuses to start unless this file and `results/real/rung1/clean_labels.csv` are committed and unmodified, and it
refuses to run a second time (`results/real/rung1/RUN_DONE`). **Real data. The reference is a FITTED lattice, not CAD.**

## 1. Question

Does the frozen v3 verifier stay quiet on real SEM images of structures with no topological defect? In other words,
how often does it call KILLER on a real tile whose topology matches the design?

This measures false alarms only. With no defects present, it says nothing about recall.

## 2. Data (downloaded by `scripts/real/download_rung1.py`, md5-checked, kept in `data/real/`, git-ignored)

| Group | Source | Licence | What | Pixel size |
|---|---|---|---|---|
| G29 | Yang, Klughammer, Barth, Tanenbaum, Dekker (ACS Nano 2023), doi:10.5281/zenodo.8060099, `Gla_29_Pd_1.zip` | CC-BY-4.0 | FIB-milled zero-mode-waveguide hole arrays in a Pd film (array v2): 32 top-view images, 512 x 768 px, Thermo Helios G4 CX, 15 kV, TLD/SE | 5.14-7.71 nm/px, FEI TIFF metadata |
| G01 | same record, `Gla_0127_14.zip` | CC-BY-4.0 | array v1, 10 kV: `z_110_overview_002` (11.24 nm/px), `z80_d_220-280_18` (10.79 nm/px), `center_overview_001` (20.75 nm/px) | FEI TIFF metadata |
| INGAN (spot-check) | Athanasiou et al., doi:10.5281/zenodo.3747155, Figure 2(b) and 2(d) top views | CC-BY-4.0 | InGaN nanohole arrays (published JPEG figure panels) | about 2.3 nm/px, from the 200 nm scale bar |

**Selection rule** (`select_zmw`; applied to file names and metadata only):
- **G29:** files named `2022_03_15_*` with stage tilt 0.
- **G01:** `*overview*` or `*220-280*` files with tilt 0 and pixel size ≤ 30 nm.
- **Excluded:** 37 tilted (52°) cross-sections; 21 fields with a single well or two wells; `full_array_overview` (74 nm/px, holes about 3 px).

The list is in `results/real/rung1/excluded.csv`. The manifest (`results/real/rung1/manifest.csv`) records, for every image, its sha256, pixel size and its source, detector, kV, beam current, dwell time, field width, tilt, clipping and licence.

**Intensities** are raw 8-bit grey levels. The dose in e⁻/px is unknown. Module 1 fits the grey-level noise model as it does for any layer.

**Display overlay.** Every G29 image carries a burned-in yellow centre cross (61 px where R ≠ B) at rows 241–271, cols 369–399. Those pixels keep their stored grey value; nothing is inpainted. Tiles touching the overlay are excluded from calibration and from the primary denominator. They are still run and reported on their own.

## 3. Frozen verifier (no change)

- `results/v3/calib/decisions.json` (sha256 `73ae3ba0…0180ac`, last changed in commit 5ca317b): the `pipeline_cfg` and `event_rule_cfg` exactly as frozen.
- Called through `verify_patch_v3` with `Window(256, 16)`.
- Module 1 uses `calibrate()` defaults (sigmas 1.5 / 4.0 px, ridge 1e-3, registration on).

No setting is chosen from real data. The only allowed post-run change is a fix for a crash (an exception). Any such fix is disclosed and the run is repeated with identical settings.

## 4. Reference: FITTED, not CAD (`src/nanovate/real/`)

A hole array is modelled as a lattice of equal circular holes (dielectric) in a continuous film (conductor). A lattice of equal circles cannot represent a missing, merged or extra hole, so it cannot absorb a topological defect. It **can** absorb or misstate hole size and shape.

| Part | G29 | G01 |
|---|---|---|
| Lattice basis | Fitted jointly on the three calibration images only: positions 0, 16, 31 in the sorted list (`…120_dset_200_060`, `…240_dset_194_075`, `…280_dset_260_047`). Result: 1513 x 1523 nm, rotation < 0.1° | Fitted on the calibration region only: tile rows 0–1 of `z_110_overview_002`. Result: 1521 x 1520 nm, rotation 0.26° |
| Hole diameter | **Design value** from the file name (`dset`, the FIB set diameter, nm) | **Fitted** on the same calibration region: 234 nm. Known weakness: the G01 arrays are diameter series, so hole size changes by column (100–280 nm) and one diameter matches only some columns |
| Translation | Registered per image from that image's own hole centres (circular median modulo the lattice, robust to a missing or extra hole). Two parameters; real images share no coordinate frame | Same |

**Module-1 stand-in scribe patches** (3 per group):
- **G29:** in each calibration image, the overlay-free 256 x 256 tile with the most fully contained reference holes. All three are tile (0, 0).
- **G01:** tiles (0, 0), (0, 2), (0, 4) of `z_110_overview_002`.

**Evaluation tiles** (non-overlapping 256 x 256 from the top-left corner):
- **G29:** all tiles of the 29 non-calibration images (174, of which 58 touch the overlay).
- **G01:** tile rows 2–3 of `z_110_overview_002` (12), all of `z80_d_220-280_18` (6) and all of `center_overview_001` (24).

**INGAN spot-check** (descriptive only; not clean, see §6):
- **Calibration:** row 0 of each image.
- **Evaluation:** row 1 (3 tiles per image).
- **Disclosed weakness:** one tile row holds only one row of holes, so the lattice and diameter are fitted on the whole image.

## 5. Scale arms (decided at Stop Point 1; written before the run)

- **Pitch-matching: infeasible, not run.** The design pitch is about 1.5 µm with holes of 175–300 nm (pitch/diameter about 5–8, against the testbed's 36/16 = 2.25). Scaling the pitch to 36 px shrinks a G29 image to about 99 x 66 px, so no 256 tile fits. Holes become about 5 px, and about 3 tiles would remain in total.
- **Arm C (primary): native pixels.** No resampling; all px settings frozen. Holes are 10–56 px (testbed vias: 14–18 px).
- **Arm B (secondary): diameter-matched.**
  - Each image is resampled by s = 16 px / (reference diameter in px): order 1, anti-aliased when s < 1. The reference is re-registered on the resampled image.
  - G01 factors: `z_110` x0.768, `z80` x0.737, `center_overview` x1.418 (upsampled).
  - G29 is excluded from arm B: after resampling its images are about 180 px tall, under one tile.
  - Calibration uses the same three G01 tile positions in the resampled frame. Evaluation tiles exclude any resampled `z_110` tile overlapping the native calibration rows (46 tiles).

## 6. Ground truth for "clean"

`results/real/rung1/clean_labels.csv` holds a visual label per image, judged by Claude (AI assistant) on every full image at native resolution with the fitted reference overlaid, **before any verifier run**. It is not an expert label.

- **Clean** = every lattice hole present, none merged, no extra hole.
- **All 35 ZMW images: clean.** Debris particles, dark contamination rectangles, rough rims and redeposition are nuisance, not defects.
- **Both INGAN panels: not clean.** They show small dark pits (about 30–60 nm) between the holes, which are extra holes relative to the design lattice. They are reported as a descriptive spot-check: there, a KILLER call is arguably correct.

## 7. Metrics and pre-registered targets

| Metric | Population | Target | Met if |
|---|---|---|---|
| **False alarm: KILLER (any KILLER_*) on clean tiles** | Arm C, G29 + G01 clean, overlay-free evaluation tiles (116 + 42 = 158 tiles, 32 images) | **≤ 5%** | point estimate ≤ 5% (CI reported; if the CI's upper end is above 5%, that is stated next to the result) |
| Same, secondary | Arm B, G01 (46 tiles, 3 images) | ≤ 5% | as above; secondary |
| NEAR_MISS rate on clean tiles | as above | none | reported |
| Calibration error | Every evaluation tile, within the window: ‖T(C) − I‖/‖I‖, ‖T(C) − I‖/‖I − mean I‖, and the contrast-relative version with the Module-1 noise model's expected noise energy subtracted | none yet | reported per group, plus in-sample values on the calibration tiles |
| Alignment error | (a) Each detected hole's centre to the nearest fitted-reference centre (nm), evaluation images; (b) Module 3's residual shift per tile (px) | none yet | reported |
| Run time | Calibration ms per group; ms per tile (total and per module), this CPU | none | reported |

**Statistics.** Each rate is a mean over tiles. Its 95% CI comes from a percentile bootstrap over **images** (tiles of one image are resampled together; 1000 resamples, seed 0), because tiles from one image are not independent. If there are zero events, the one-sided 95% upper bound 3/n_images is also given. Sample counts (tiles and images) sit next to every number.

**Failure examples.** The first five clean, overlay-free arm C tiles called KILLER (sorted by group, image, tile) are saved as PNGs with their verdict reason (Δβ, located events, bottleneck, SDF energy).

**Reported regardless.** Overlay tiles and the INGAN spot-check (KILLER rate on not-clean tiles) are reported separately and are not part of the target.

## 8. What this can and cannot tell us (written before the run)

- **Can:** whether real SEM contrast, charging halos, grain, debris and rough FIB edges make the frozen rule invent topology changes on structures that have none.
- **Cannot:**
  - recall;
  - behaviour on lines, gaps or bridges (a hole array has no conductor pairs, so the v3 merge rule barely fires);
  - behaviour at the 14 nm scale the verifier was built for;
  - behaviour on true CAD.
- **Confounds:** a FITTED reference (and, for G01, a diameter-series mismatch); a visual "clean" label; FIB-milled metal film rather than lithographic resist; and real holes that are the inverse polarity of testbed family C (conductor pillars).
