# Track C Rung 1c pre-registration: near-miss thresholds re-selected under a synthetic edge halo, then ONE real repeat

*Part A is committed before any halo patch is rendered or any threshold is computed. Part B is committed before the real
run. The run guard is the same as Rungs 1/1b (committed files, run once: `results/real/rung1c/RUN_DONE`).*

**Why.** In Rung 1b, the template clamp removed most fake topology, but 57% of clean G29 tiles became NEAR_MISS
(`docs/REAL_DATA_RUNG1B.md`). The near-miss gate (d*, e*, z*) was selected on a testbed whose edge bloom is narrow
(σ_edge 1.3 nm). Real holes have a wide bright SE halo. Hypothesis: thresholds selected with halo-stressed nuisance in the
constraint set stop firing on real rims, at some cost in synthetic near-miss recall.

**Disclosure.**
- The motivation comes from the Rung 1/1b real runs. No real image or real per-tile number is read by the selection code.
- Rung 1b's committed `results/real/rung1b/tiles.csv` contains per-tile bottleneck and SDF energy. It is not opened to evaluate any candidate threshold before the run.
- Halo values are chosen a priori below, not fitted to real images.

## Part A: halo model and the selection rule (synthetic, Manhattan val only)

**Base pipeline.** Rung 1b frozen (`results/real/rung1b/val/decisions.json`): v3 rule, plus C1 guard, C2 template clamp and C3 bounded alignment. Only d*, e* and z* are re-selected. τ, the field, the event rule and the killer path are unchanged, so **no KILLER verdict can change**.

**Halo stress cells** (no new simulator code). The existing SEM model's bloom term `a2 · s·tanh(|∇(G_σedge * W)|/s)` is made wider and stronger.

| Cell | σ_edge (nm) | a2 | Meaning |
|---|---|---|---|
| nominal | 1.3 | 0.9 | testbed as is |
| H1 | 3.0 | 0.9 | 2.3x wider bloom |
| H2 | 3.0 | 1.8 | wider and 2x stronger |
| H3 | 5.0 | 0.9 | 3.8x wider (beyond Module 1's σ₂ = 4 px feature) |
| H4 | 5.0 | 1.8 | widest and strongest |

The values are a priori round multiples of the nominal. Limitation: the testbed bloom is symmetric across the edge, while the real halo sits on the film side.

The halo cells run at nominal LER 1 nm and grain 15%, at doses 20/50/200. The scribe patches of each halo cell are rendered with the same halo (calibration is per layer).

**Selection set.**
- v3's selection set: Manhattan val layouts 0–7 x 2 dies x 6 kinds x doses 20/50/200 x (LER 0.5/1.0/1.5 nm x grain 0/15/30%) = 2,592 patches;
- **plus** the 4 halo cells x the same layouts, dies, kinds and doses = 1,152 patches.

**Rule.** v3's robust procedure, unchanged (`experiments/v3_select.py::near_miss`): maximise near-miss (notch + filament) recall at the nominal cell, subject to **NEAR_MISS ≤ 5% of nuisance patches in every cell**, now including H1–H4. "Zero" means no killer event under the base pipeline. The grids come from quantiles, as in v3. If no grid point is feasible, the near-miss rule is switched off (d* = ∞) and that is reported.

**Control (reported, not adopted).** The same procedure on the 2,592 patches without halo cells, on the 1b base. It shows how much of any change is due to the halo cells.

**Reported cost (no kill condition).** On held-out val layouts 8–23 (not in the selection set) x 2 dies x 6 kinds at dose 50:
- near-miss recall and nuisance NEAR_MISS rate at the nominal cell and in each halo cell;
- old (v3) vs new thresholds, with 95% bootstrap CIs and n.

Frozen to `results/real/rung1c/val/decisions.json`.

## Part A result (synthetic; `results/real/rung1c/val/`)

| | d* | e* | z* |
|---|---|---|---|
| Previous (v3 / 1b) | 0.0780 | 2.575 | 5.461 |
| Control: re-selected without halo cells, 1b base | 0.0780 | 2.575 | 5.461 (identical, as expected) |
| **Adopted: re-selected with halo cells H1–H4** | **0.0846** | **2.545** | **5.374** |

- **Feasible.** The guard never triggered on the 3,744 selection patches.
- **Held-out val (layouts 8–23, dose 50, n = 32 nuisance and 64 notch/filament per cell):**
  - nuisance NEAR_MISS is 0–3% in every cell (nominal and H1–H4), under both old and new thresholds;
  - near-miss recall at nominal is 62.5% for both;
  - H2 is 46.9% → 48.4%.
- **Reading, written before the real run.** The synthetic halo barely stresses the gate. Module 1, calibrated on halo-rendered scribe patches, absorbs a symmetric wider bloom. So the testbed halo does not reproduce whatever drives the real rim NEAR_MISS flags. The **expected real effect is small**. A plausible untested driver is the ~40 nm design-vs-real hole-diameter offset (a systematic CD bias), which the testbed never generates.

## Part B: the real repeat (written after Part A's result, committed before the real run)

- **Pipeline:** Rung 1b frozen (guard, clamp, bounded alignment), with (d*, e*, z*) replaced by the adopted values above. Nothing else changes. KILLER and UNVERIFIED verdicts are therefore identical to Rung 1b by construction; only NEAR_MISS vs NUISANCE can move.
- **Populations, references, calibrations, labels:** exactly Rung 1b's (orig: 158 clean overlay-free tiles in 32 images; fresh held-out: 68 film-only tiles in 17 images). `python -m experiments.real_rung1c --run`, once; paired tile by tile with Rung 1b.

**Target (new, pre-registered here):**

| Metric | Target | Notes |
|---|---|---|
| **NEAR_MISS on clean overlay-free tiles** | **≤ 5%** (the val near-miss budget) | judged separately on orig and fresh. Rung 1b values for reference: orig G29 56.9%, fresh 8.8% |

KILLER, coverage and NUISANCE are reported again, with the Rung 1b targets unchanged (KILLER ≤ 5% and coverage ≥ 80%). 95% CIs come from a bootstrap over images (1000 resamples, seed 0).
