# Track C Rung 1d pre-registration: systematic CD offset in the testbed, CD-corrected SDF energy, then ONE real repeat

*Part A is committed before any CD-offset patch is rendered or any candidate is scored. Part B is committed before the real
run. Same run guard as Rungs 1/1b/1c (`results/real/rung1d/RUN_DONE`).*

**Why.** Rung 1c (`docs/REAL_DATA_RUNG1C.md`) left 48% of clean G29 tiles flagged NEAR_MISS. The suspected driver is a
systematic CD offset: the real holes are about 40 nm wider than their design diameter, so the SDF energy is large at
every rim. The testbed never generates such an offset, so no gate re-selected on it could learn to ignore one.

**Disclosure.** The motivation comes from the real runs. No real image or real per-tile number is read by any selection
code. The committed Rung 1b/1c `tiles.csv` files are not used to evaluate candidates. The constants below are a priori.

## Part A (synthetic, Manhattan val only)

**1. CD offset in the testbed** (new `testbed/cdbias.py`; the existing generators stay byte-identical).
- The true wafer W from `make_wafer` (LER plus defect, label-checked) becomes **W′ = grow(W, b)**:
  - b > 0: add every dielectric pixel within b nm of conductor;
  - b < 0: remove every conductor pixel within |b| nm of dielectric (exact Euclidean distance transform at 1 nm/px).
- W′ is rendered with the same random streams as `make_sample`.
- The label is recomputed from W′. A patch whose topology the offset itself changes (vs. its kind) is **excluded and counted**.
- The Module 1 scribe patches of the same cell are rendered with the same b (it is a process bias).
- **CD cells:** b ∈ {−3, −1.5, +1.5, +3} nm, at nominal LER 1 nm and grain 15%, doses 20/50/200.

**2. Candidate: CD-corrected SDF energy** (opt-in flag on the Rung 1b robust pipeline).
- **Bias estimate.** Module 1 estimates one per-layer bias b̂ from the scribe patches only. For each b on the grid −8 … +8 px in 0.5 px steps, it refits Module 1 on the scribe patches with CAD grown by b and computes the residual Σ(y − shift(T))². b̂ is the minimiser; ties go to smaller |b|.
- **Use.** In verification, only the SDF energy (and its peak-z) is computed against grow(CAD, b̂), with the template from the refit at b̂ (same clamp). The topology field, the event rule and the killer path stay against the **design** CAD. No KILLER or UNVERIFIED verdict can change; only NEAR_MISS vs NUISANCE.

**3. Selection.** For **each** variant (V0 = energy vs design CAD; V1 = CD-corrected energy), (d*, e*, z*) is re-selected with v3's robust procedure (`v3_select.near_miss`, unchanged). The selection set is the Rung 1c set (2,592 v3-cell patches + 1,152 halo-cell patches) **plus** the 4 CD cells x val layouts 0–7 x 2 dies x 6 kinds x 3 doses (1,152 patches). The constraint is NEAR_MISS ≤ 5% of nuisance in every cell, including the CD cells.

**4. Decision rule.** Measured on held-out val: layouts 8–23 x 2 dies x 6 kinds at dose 50, in 9 cells (nominal, H1–H4, b = ±1.5, ±3).
- Adopt the variant with the higher **pooled held-out near-miss (notch + filament) recall** at its own selected thresholds. Ties within 1 percentage point go to **V0** (simpler).
- Reported for both variants: nuisance NEAR_MISS per held-out cell, near-miss recall per cell (95% bootstrap CIs, n), the b̂ recovered in each CD cell against the true b, the number of offset-excluded patches, and the calibration time with the b-grid.

Frozen to `results/real/rung1d/val/decisions.json`.

## Part B: the real repeat

*(added after Part A's result, committed before the real run)*
