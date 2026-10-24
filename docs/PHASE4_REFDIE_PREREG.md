# Phase 4 pre-registration: reference-die mode vs CAD mode on synthetic data (Track C Rung 3 prerequisite)

*Committed before the reference-die code is written or run. Synthetic data only.*

**Question.** Public SEM defect datasets almost never ship with CAD. Can the verifier use a clean neighbouring die as its
structural reference instead of CAD, and what does that cost in killer F1?

**Kill condition** (unchanged from `docs/NEXT_PATHS.md` Phase 4): if reference-die mode loses **more than 10 F1 points
against CAD mode** on synthetic data, real-data work needs CAD access. In that case, pause Rung 3 and look for a data partner.

## Reference-die mode (all choices a priori; nothing is tuned)

1. **Reference dies.** The two clean dies the die-to-die baseline already uses (die ids 900 and 901, kind = nuisance, same layout, dose and seed; `experiments/harness.py`).
2. **Pseudo-CAD**, from the reference images only:
   - align die 901 to die 900 (Module 3's sub-pixel phase correlation, `estimate_shift`, upsample 20) and average them;
   - Gaussian smoothing σ = 1.5 px; threshold at Otsu;
   - remove conductor specks and fill dielectric holes smaller than 10 px (both below the testbed's smallest feature, 14 nm).

   Conductor polarity = brighter than the threshold. That matches the testbed. On real data the polarity would have to be given; this is a known limitation.
3. **Calibration.** Module 1 is fitted on (pseudo-CAD, each aligned reference image), with no scribe patches and no CAD.
4. **Verification.** The frozen v3 verifier (`results/v3/calib/decisions.json`, `verify_patch_v3`, Window(256, 16)) runs with the pseudo-CAD in place of CAD. τ, the event rule and d*, e*, z* are unchanged.

**Variants:**
- **R2 (primary):** two reference dies, as above.
- **R1 (secondary):** reference die 900 alone.
- **R-oracle (diagnostic):** the true wafer W of die 900 used as the reference. This is the best any segmentation could do, so the R2 → R-oracle gap is segmentation loss, and the R-oracle → CAD gap is the cost of the reference die's own LER and offset.

**CAD mode (comparison).** The frozen v3 verifier with true CAD and Module 1 calibrated on the testbed's scribe patches, exactly as in v3.

## Evaluation set (validation, but not used by any v3 selection)

Val layouts **8–23** (v3 selected its rule on layouts 0–7, so neither mode is in-sample) x dies 0–1 x 6 kinds:

| Population | Dose (e⁻/px) | Patches |
|---|---|---|
| Manhattan | 20 | 192 |
| Manhattan | 50 | 192 |
| Manhattan | 200 | 192 |
| Manhattan, noise-free (reference dies noise-free too) | 50 | 192 |
| via (zero-shot) | 50 | 192 |
| curvy (zero-shot) | 50 | 192 |

Total: **1,152 patches**, the same patches for every mode. No test data is generated.

## Metrics

- **Primary:** killer F1, pooled over all 1,152 patches. ΔF1 = F1(CAD) − F1(R2), with a 95% paired bootstrap CI (1000 resamples, seed 0, patches resampled jointly). **Kill if ΔF1 > 0.10** (point estimate). The CI is reported, and if it straddles 0.10 that is stated next to the result.
- **Reported:**
  - the same per population, and for R1 and R-oracle;
  - per-kind rates (nuisance suppression, recall of bridge / open / particle, notch / filament called killer);
  - pseudo-CAD quality: the share of patches whose pseudo-CAD has different topology from the true CAD on the central window;
  - time per patch for building the reference.

Results go to `results/phase4/` and `docs/PHASE4_REFDIE.md`.
