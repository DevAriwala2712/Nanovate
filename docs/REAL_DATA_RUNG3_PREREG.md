# Track C Rung 3 pre-registration: reference-cell mode on a real 90 nm chip with deliberate cell swaps (Puschner et al.)

*Committed before the verifier is run on any image of this dataset. Run once: `python -m experiments.rung3_puschner --run`
(refuses otherwise, or a second time). **REAL DATA.***

## Data

- **Source:** Puschner, Moos, Becker, Kison, Moradi, Paar, "Red Team vs. Blue Team: A Real-World Hardware Trojan Detection Case Study Across Four Modern CMOS Technology Generations", IEEE S&P 2023.
- **Files:** doi:10.17617/3.396Q7I (Edmond), **CC-BY-4.0**. Only the **90 nm** chip's 2,070 backside SEM tiles (4096 x 4096, 8-bit, no SEM metadata), the abstracted GDSII and the MIST stitching positions were downloaded and MD5-checked (`scripts/real/download_rung3.py`). Data stays in `data/real/` (git-ignored).
- **GDSII (abstracted):** 453,850 standard-cell boxes of 467 anonymised types (7 filler types: 230,888 fillers; 222,962 functional cells), all 3.12 µm tall on a **2.52 µm row pitch** (573 rows), in 4 orientations.
- **Ground truth:** 10 cells were swapped for different same-size cells (4 filler, 6 functional). **The positions are withheld** by the authors (available on request; contacting them is the user's decision). **Recall therefore cannot be measured here.**

## Registration (done before this file; no verifier involved)

- **Coarse:** a 1/32 mosaic of all tiles was cross-correlated with the GDS functional-cell density map over 8 rotation/mirror combinations and a scale of 10–16 nm/px. Result: **rotate 90° + mirror (a backside view), 12.23 nm/px** (correlation 0.153, against ≤ 0.043 for every other orientation).
- **Consistency check:** 573 GDS rows over the registered core give 206 px per row, matching the bright row-rail period measured by FFT in every sampled core tile (204.8 px). **12.23 nm/px is a registration estimate, not stated metadata.**
- **Fine:** an affine correction fitted on every 4th tile, plus a per-tile residual offset. The offset comes from correlating bright-contact density against the registered functional-cell mask (1/4 resolution, ±160 px search). 1,025 core tiles have median score 0.33. Tiles with score < 0.2 or at the search limit use the median of their neighbours (149 tiles) or the global median (957 tiles, all outside the cell core).
- **Visual check:** windows of one cell type look alike, and the cell box falls between the row rails (`data/real/puschner/work`, not committed).

## Reference-cell mode (`src/nanovate/refcell.py`, `experiments/rung3_puschner.py`; all choices fixed here)

- **Working scale x0.5** (24.5 nm/px): a row is about 103 px, so a 256 x 256 window holds the row plus about 76 px of each neighbouring row.
- **Cell box:** the GDS box trimmed by 0.30 µm on each row side (to the 2.52 µm pitch). Cells wider than 224 px get several windows (step 224 px).
- **Home tile:** each window must lie entirely inside one tile, after that tile's offset; the most central such tile is used. Windows that fit no tile (5.4%; the 10% tile overlap is smaller than a window) are not verified and are counted.
- **Groups:** cell type x orientation. **Reference:** for each group with ≥ 4 instances, **15 instances chosen by a seeded draw** (seed 0; all if fewer). Their windows are aligned and the **median** taken. The pseudo-CAD is Otsu of the smoothed median (σ 1.5 px, 10 px cleanup). Module 1 is calibrated on (pseudo-CAD, 3 medians of interleaved subsets) with the frozen C1 guard and C2 clamp.
- **Verification:** each window of each instance is aligned to its group median and **spliced** (instance pixels inside the cell box, median outside). The **latest frozen verifier** then runs: Rung 1d, i.e. the v3 rule, C1/C2/C3, and near-miss gate d* 0.0798, e* 2.535, z* 5.326. Nothing is tuned.
- **Dry-run coverage** (no verifier): 427,519 of 453,850 instances verifiable (94.2%; 218,048 fillers, 209,475 functional) in 737 referenced groups, with 485,920 windows.

## Metrics and targets

Instance-level: an instance is KILLER if any of its windows is. 95% CIs come from a bootstrap over **home tiles** (1000 resamples, seed 0).

| Metric | Population | Target |
|---|---|---|
| **KILLER rate** | all verified instances (at most 10 can be true swaps, so this is essentially the false-alarm rate) | **≤ 5%** |
| **NEAR_MISS rate** (instances with no KILLER window) | same | ≤ 5% |
| **Coverage** = verified instances / all 453,850 GDS instances | — | **≥ 80%** |

**Reported:**
- the same rates for functional and filler cells separately;
- the **topology-change rate** (Δβ ≠ 0 in any window). The frozen v3 rule calls a vanished component NEAR_MISS rather than KILLER (`tests/test_refcell.py`), so Δβ is the more direct cell-swap signal;
- calibration reliability per reference, time per window;
- a **candidate list**: the top 200 verified instances by (KILLER, |Δβ₀| summed over windows, topology change, max SDF energy), saved as `candidates_top200.csv` with GDS coordinates, for later checking against the authors' positions.

**Interpretation rules** (written before the run):
- A low KILLER rate is evidence the reference-cell mode is quiet on real cells. A high rate means it isn't.
- The candidate list can be called a *detection* only if the positions are obtained and match. Until then it is a ranked list, not a result.
