# Track C real-wiring false-alarm test (M2) pre-registration

*Committed before the verifier is run on any image of this dataset. Run once: `python -m experiments.real_m2 --run` (refuses
otherwise, or a second time). **REAL DATA.***

**Question.** On real Manhattan wiring, against a validated layout extraction instead of a fitted lattice, how often does
the frozen verifier call a clean tile KILLER or NEAR_MISS? This is the cleanest real false-alarm test so far.
- The structures are the testbed's own kind: tracks, gaps and vias.
- The reference is real geometry.
- The feature scale is close to the testbed.

## Data

- **Source:** "A Real-World Metal-Layer SEM Image Dataset with Partial Labels" (paper: "Towards Unsupervised SEM Image Segmentation for IC Layout Extraction"), doi:10.17617/3.HY5SYN, **CC-BY-4.0**. Downloaded and MD5-checked by `scripts/real/download_rung3.py` into `data/real/` (git-ignored).
- **Images:** 321 metal-2 (M2) images, 4096 x 3536 px, **14.65 nm/px** (stated by the authors), backscattered electrons, 15 keV, 3 µs dwell, of the logic area of a commercial 128 nm IC.
- **Manifest:** `results/real/m2/manifest.csv` (sha256, pixel size, licence, label and QC statistics per image).

**Reference.** The authors' M2 labels: tracks as SVG polygons and vias as circles, both rasterised as conductor (`src/nanovate/real/m2.py`). The authors produced them automatically and then **manually validated and corrected** them.

**What this means for interpretation:**
- The labels come from these same images, so a real defect would also be in the labels. This test measures false alarms only.
- It is a layout *extraction*, not the designer's CAD.

**Clean.** Images of a working production chip are taken as defect-free. There is no per-image visual label: there are too many images, and the labels already encode the real topology.

## Label-completeness QC (no verifier involved)

The authors warn that tracks spanning several images may be missing from a label file. An unlabelled real track would look like a false particle or bridge.

**QC rule** (`unlabelled_conductor`), fixed before any run:
- smooth the image (σ 2 px);
- threshold at the midpoint between the median inside the labels and the median outside the labels dilated by 5 px;
- flag bright components of 200 px or more lying outside that dilated label.

A tile touching a flagged component is **QC-flagged**. Dry-run statistics, computed before this file was committed: 832 of 66,144 native evaluation tiles (1.3%).

**Disclosed bias.** The QC also removes tiles with large bright debris, which could otherwise be a real verifier false alarm. Both populations are therefore reported.

## Verifier (frozen; nothing tuned on these images)

The latest frozen state, **Rung 1d**:
- the v3 event rule (`results/v3/calib/decisions.json`);
- C1 guard, C2 template clamp and C3 bounded alignment (`results/real/rung1b/val/decisions.json`), with `cd_correct` off;
- near-miss gate d* 0.0798, e* 2.535, z* 5.326 (`results/real/rung1d/val/decisions.json`, V0);
- Window(256, 16).

**Calibration (Module 1 stand-in scribe patches):** the **central 256 x 256 tile** of images 0, 160 and 320 in the sorted id list (`0000`, `0160`, `0320`). Their labels serve as the scribe CAD. These three images are excluded from evaluation.

**Arms:**
- **native (primary):** 14.65 nm/px. Tracks are about 25–35 px wide, against the testbed's 14–18 px.
- **half (secondary, pre-declared):** image and label area-averaged to x0.5 (29.3 nm/px; tracks about 12–17 px, near the testbed width). The label raster is re-thresholded at 0.5, and calibration uses the central tiles of the resampled calibration images. This directly tests whether matching the feature scale matters.

**Tiles:** every full 256 x 256 tile of the 318 evaluation images, from the top-left corner (native 208 per image; half 48 per image). Tiles whose label raster is all one class are skipped and counted.

## Metrics and targets

95% CIs come from a bootstrap over images (1000 resamples, seed 0), with n tiles and n images beside every number.

| Metric | Population | Target |
|---|---|---|
| **KILLER rate** | native, QC-passed tiles (**primary**) | **≤ 5%** |
| **NEAR_MISS rate** | same | **≤ 5%** |
| **Coverage** (not UNVERIFIED) | same | **≥ 80%** |

All three targets are also reported for the half arm, for QC-flagged tiles and for all tiles. Each target is judged as met or not met on its point estimate, with the CI shown.

Also reported: verdict breakdown, Δβ and event counts, alignment shifts and rejections, calibration contrast and leave-one-out values, and time per tile. After the run, up to 5 KILLER tiles from the primary population (first by image id, then tile) are saved as PNGs with their verdict reason (Δβ, bottleneck, SDF energy). These are illustration only and change nothing.
