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

## Part A result (synthetic; `results/real/rung1d/val/`)

**Decision: V0 is adopted, and the CD-corrected energy (V1) is rejected.** Pooled held-out near-miss recall (467 notch/filament patches in 9 cells): V0 55.0% [50.1, 60.0], V1 37.5% [33.6, 42.0].

| | d* | e* | z* |
|---|---|---|---|
| Rung 1c gate | 0.0846 | 2.545 | 5.374 |
| V0 (adopted; selected with the CD cells added) | 0.0798 | 2.535 | 5.326 |
| V1 (not adopted) | 0.0798 | 2.629 | 4.906 |

- **Bias estimate:** accurate in the CD cells (true −3 / −1.5 / +1.5 / +3 nm → estimated −3.5 / −1.5 / +1.5 / +3.0 px). In the **halo** cells (no CD offset) it returns +2 to +3 px, reading a wide bloom as wider conductor. As a result, V1 raises held-out nuisance NEAR_MISS there (H1 6%, H3 9%, H4 12.5%, n = 32 each). That is why V1 loses.
- **The synthetic CD offset does not stress V0's gate either:** held-out nuisance NEAR_MISS is 0% in all four CD cells. Module 1, calibrated on scribe patches with the same offset, absorbs it into the template.
- **Cost on synthetic near-misses:** recall in the CD cells is low under both variants (V0: −3 nm 7%, +1.5 nm 16%, +3 nm 0%).
- **Excluded:** 273 selection and 207 held-out patches, where the offset itself changed the topology (mostly notch/filament/particle at ±3 nm on 14 nm features).
- The guard never triggered.

**Reading, written before the real run.** Neither variant gives a new mechanism for real rims. The adopted change is only another small shift of the gate (e* 2.545 → 2.535, z* 5.374 → 5.326; d* 0.0846 → 0.0798). Because e* and z* got slightly *lower*, real NEAR_MISS may stay flat or rise slightly.

## Part B: the real repeat (written after Part A's result, committed before the real run)

- **Pipeline:** Rung 1b robust config (guard, clamp, bounded alignment; `cd_correct` **off**), with (d*, e*, z*) = V0's values above. KILLER and UNVERIFIED verdicts are identical to Rungs 1b/1c by construction.
- **Populations, references, calibrations, labels:** exactly Rung 1b/1c's. `python -m experiments.real_rung1d --run`, once; paired tile by tile with Rung 1c.
- **Target (unchanged from 1c):** NEAR_MISS on clean overlay-free tiles ≤ 5%, judged separately on orig and fresh. Also reported: KILLER (≤ 5%), coverage (≥ 80%), NUISANCE. 95% CIs come from a bootstrap over images (1000 resamples, seed 0).
- **Diagnostic, computed after the run and labelled post-hoc:** the per-layer bias b̂ that V1 *would* have estimated on the real calibration tiles. It is reported only to see whether the real offset is visible to the estimator. It changes no verdict.
