# Track C Rung 1c: near-miss gate re-selected under a synthetic edge halo, ONE real repeat

**REAL DATA. The reference is FITTED, not CAD.**
- Pre-registration: `docs/REAL_DATA_RUNG1C_PREREG.md` (Part A at 2b4ab09; Part B, the frozen thresholds and the stated expectation at 504d047).
- Run once: `python -m experiments.real_rung1c --run` → `results/real/rung1c/`.

## 1. What changed

Only the near-miss gate (d*, e*, z*). It was re-selected with v3's own robust procedure on synthetic Manhattan val. The constraint set now also contains four **edge-halo stress cells** (bloom width σ_edge 3 or 5 nm, gain 0.9 or 1.8, against the nominal 1.3 nm / 0.9). The base is Rung 1b's frozen pipeline (guard, template clamp, bounded alignment).

| | d* | e* | z* |
|---|---|---|---|
| Rung 1b (= v3) | 0.0780 | 2.575 | 5.461 |
| Rung 1c (with halo cells) | 0.0846 | 2.545 | 5.374 |

**Synthetic held-out val** (layouts 8–23, dose 50; 32 nuisance and 64 notch/filament per cell):
- nuisance NEAR_MISS is 0–3% in every cell, under the old and the new gate;
- near-miss recall at nominal is 62.5% under both;
- the halo cells sit at 47–75%, almost unchanged.

**Stated before the run:** the synthetic halo barely stresses the gate (Module 1 absorbs a symmetric bloom), so the expected real effect is small.

## 2. Results vs targets

Bootstrap over images, 1000 resamples, seed 0.

| Population | NEAR_MISS on clean tiles (target ≤ 5%) | Rung 1b | KILLER | Coverage |
|---|---|---|---|---|
| **fresh (held out, film only)**: 68 tiles, 17 images | **4.4% [0, 10.3]**: **met** on the point estimate; the CI's upper end (10.3%) is above 5% | 8.8% | 0% | 100% |
| **orig (Rung 1 tiles)**: 158 tiles, 32 images | **35.4% [21.4, 52.3]**: **not met** | 41.8% | 13.3% [5.7, 23.6] (unchanged) | 73.4% (unchanged) |
| — orig / G29: 116 tiles, 29 images | 48.3% [37.1, 59.5] | 56.9% | 18.1% | 100% |
| — orig / G01: 42 tiles, 3 images | all UNVERIFIED (guard) | — | — | 0% |

**Paired with Rung 1b (verified tile by tile):**
- Every KILLER and UNVERIFIED verdict is identical, as the design guarantees.
- The only moves are NEAR_MISS → NUISANCE: 10 of 66 orig tiles and 3 of 6 fresh tiles.
- No tile moved the other way.

**Overall:**
- **Fresh** now meets every pre-registered target on its point estimates: KILLER 0%, coverage 100%, NEAR_MISS 4.4%.
- **Orig** still fails all three (KILLER 13.3%, coverage 73%, NEAR_MISS 35%).

## 3. What this tells us

1. **The halo hypothesis is mostly falsified as tested.** Re-selecting the gate under a symmetric synthetic halo moved it by under 10%, and G29's rim NEAR_MISS fell only from 57% to 48%. The testbed's halo doesn't reproduce what triggers the gate at real rims.
2. **The remaining G29 near-miss flags are a reference problem more than a threshold problem.** The real holes are about 40 nm wider than their design diameter, so the SDF energy at every rim is large by construction. The testbed never has a systematic CD offset between CAD and wafer, so no amount of re-selection on it will learn to ignore one. A further threshold change would just be tuning by another name.
3. **On held-out film**, the full stack (guard, clamp, bounded alignment, re-selected gate) is quiet: 0 killers and 3 near-misses in 68 tiles. That is the first real-data population to meet all three targets. It is film only, so rims are untested there.

## 4. What it does NOT tell us

- No recall (no defects).
- Fresh contains no rims (each well sits under the burned-in cross).
- Orig informed the design of every Rung 1b/1c change.
- The near-miss recall cost on synthetic val is about zero, but that says nothing about real near-misses.
- Labels are visual, by the assistant; the reference is FITTED.

## 5. Next (not done; each change to be chosen on synthetic val only)

- **Add a systematic CD offset to the testbed** (wafer features uniformly wider or narrower than CAD by a few nm). Then decide on synthetic val whether the SDF energy should be measured against a CD-corrected template, for example with a per-layer CD bias estimated by Module 1 from the scribe patches. This targets the mechanism Rung 1c points to.
- **Or stop tuning on these images.** Every change so far has been motivated by them. Rung 2 (test chip with true CAD and known defects) is the clean next test, including whether the clamp hides real defects.
