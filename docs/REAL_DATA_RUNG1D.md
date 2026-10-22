# Track C Rung 1d: systematic CD offset in the testbed, CD-corrected SDF energy (rejected on val), ONE real repeat

**REAL DATA. The reference is FITTED, not CAD.**
- Pre-registration: `docs/REAL_DATA_RUNG1D_PREREG.md`; Part A at cf6f258; Part B, the frozen decision and the stated expectation at 16a9c17.
- Run once: `python -m experiments.real_rung1d --run` → `results/real/rung1d/`.

## 1. What was built and decided on synthetic val

**Testbed CD offset** (`src/nanovate/testbed/cdbias.py`).
- The true wafer is grown or shrunk by b nm (exact distance transform) and rendered with the same random streams. The label is recomputed from the offset wafer.
- b = 0 reproduces the testbed bit for bit (tested).
- The scribe patches get the same offset.
- CD cells: b = ±1.5 and ±3 nm. The offset itself changed the topology in 273 of 4,896 selection patches and 207 of 1,728 held-out patches; those were excluded as pre-registered.

**Candidate V1, CD-corrected SDF energy.**
- Module 1 estimates a per-layer bias b̂ from the scribe patches (grid −8 to +8 px); the energy is measured against CAD grown by b̂.
- The topology and the killer path stay on design CAD (verified: identical killer flags on every val patch).

**Gate re-selection for both variants** (v3 procedure; constraint cells = v3 cells + 1c halo cells + CD cells). Held-out val (layouts 8–23, dose 50, 9 cells):

| | Pooled near-miss recall (n = 467) | Pooled nuisance NEAR_MISS (n = 276) | Gate (d*, e*, z*) |
|---|---|---|---|
| **V0, energy vs design CAD (adopted)** | **55.0% [50.1, 60.0]** | 0.7% [0, 1.8] | 0.0798, 2.535, 5.326 |
| V1, CD-corrected energy | 37.5% [33.6, 42.0] | 4.0% [1.8, 6.5] | 0.0798, 2.629, 4.906 |

- The bias estimate is accurate in the CD cells (true −3/−1.5/+1.5/+3 nm → −3.5/−1.5/+1.5/+3.0 px).
- **But** in the halo cells (no offset) it returns +2 to +3 px. It reads a wide bloom as wider conductor, which raises nuisance NEAR_MISS (H4: 12.5%). That is why V1 lost.
- Under V0, the synthetic CD offset produced 0% nuisance NEAR_MISS. Module 1, calibrated on offset scribe patches, absorbs it.

## 2. Real results vs targets (NEAR_MISS ≤ 5% on clean overlay-free tiles; bootstrap over images)

| Population | NEAR_MISS 1d | 1c | 1b | KILLER | Coverage | NEAR_MISS target |
|---|---|---|---|---|---|---|
| **fresh (held out, film only)**: 68 tiles, 17 images | **7.4% [0, 14.7]** | 4.4% | 8.8% | 0% | 100% | **not met** (met in 1c) |
| **orig**: 158 tiles, 32 images | **41.1% [25.8, 58.1]** | 35.4% | 41.8% | 13.3% | 73.4% | **not met** |
| — orig / G29: 116 tiles, 29 images | 56.0% [45.7, 67.2] | 48.3% | 56.9% | 18.1% | 100% | — |

**Paired with Rung 1c:**
- Every KILLER and UNVERIFIED verdict is identical.
- The only moves are NUISANCE → NEAR_MISS: 9 orig tiles and 2 fresh tiles. No tile moved the other way. This is exactly the effect of a gate set slightly lower (e* 2.545 → 2.535, z* 5.374 → 5.326), as Part B warned before the run.

**Post-hoc diagnostic (pre-declared, changes no verdict):** the bias V1 would have estimated on the real calibration tiles.

| Group | b̂ | Physical |
|---|---|---|
| G29 | −4.5 px | **−24 nm** film edge, so holes about 48 nm wider in diameter. Matches the ~40 nm design-vs-real offset measured independently in Rung 1 |
| FRESH | −4.5 px | −11 nm |
| G01 | −3.5 px | −39 nm (calibration unreliable anyway) |

## 3. What this tells us

1. **Rung 1d's adopted change made real false flags slightly worse, not better.** By the pre-registered rule, the 1d gate (V0) is the frozen state. Fresh no longer meets the near-miss target. That's a regression to report, not to undo after the fact. The 1c gate remains on record (`results/real/rung1c/`) as the better real result, but choosing between them now on real data would be tuning on real data.
2. **The synthetic testbed is the bottleneck for this line of work.** Three rungs in a row (halo, CD offset, threshold re-selection) found that the synthetic stressors are absorbed by Module 1 calibration and so never stress the near-miss gate. On real images, the same calibration does not absorb the rim mismatch. Re-selecting thresholds on this testbed cannot fix that.
3. **The CD estimator sees the real offset** (−24 nm on G29, matching the independent measurement). It was rejected only because, on synthetic data, it confuses wide halos with CD. A version that separates bloom width from CD (estimating both jointly from the scribe patches) is the one clearly motivated next candidate. It still has to win on synthetic val first.

## 4. What it does NOT tell us

- No recall.
- Fresh has no rims.
- Orig informed all of Rungs 1b–1d.
- References are FITTED, and labels are visual by the assistant.
- The post-hoc b̂ agreement is one number per layer, not a validation.

## 5. Recommendation

Stop tuning on these images. Every change since Rung 1 was motivated by them, and the last two moved real results by noise-level amounts in opposite directions.

Next steps:
- **Rung 2** (true CAD, known defects) is the next informative test.
- On the synthetic side, the one candidate worth a pre-registered try is a joint bloom-width plus CD estimate in Module 1.
