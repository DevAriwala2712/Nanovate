# Phase 4: reference-die mode vs CAD mode (synthetic data, val layouts 8–23)

- Pre-registration: `docs/PHASE4_REFDIE_PREREG.md` (commit c2e3601, before any reference-die code).
- Code: `src/nanovate/refdie.py`, `experiments/phase4_refdie.py`. Results: `results/phase4/`. **Synthetic data only.**

## What was built

**Reference-die mode** uses no CAD at all:
- the two clean neighbouring dies (900, 901) are aligned (sub-pixel) and averaged;
- the mean is smoothed (σ 1.5 px), thresholded at Otsu, and cleaned of specks and holes under 10 px. This gives a **pseudo-CAD**;
- Module 1 is calibrated on (pseudo-CAD, reference images) instead of scribe patches;
- the frozen v3 verifier runs against the pseudo-CAD, with no re-tuning.

Every choice was fixed in the pre-registration.

## Result vs the kill condition

Kill if reference-die mode loses more than 10 F1 points against CAD.

| Mode | Killer F1, pooled (n = 1,152) | Loss vs CAD (95% paired CI) |
|---|---|---|
| CAD (frozen v3) | 0.849 | — |
| **R2: two reference dies (primary)** | **0.803** | **4.6 points [2.3, 6.9]: kill condition NOT triggered** |
| R1: one reference die | 0.802 | 4.7 [2.4, 7.0] |
| R-oracle: the reference die's true wafer | 0.804 | 4.5 [2.7, 6.3] |

**By population** (R2 vs CAD; n = 192 each):

| Population | CAD | R2 | Loss [95% CI] |
|---|---|---|---|
| Manhattan, dose 20 | 0.773 | 0.735 | 3.8 [−3.0, 11.3] |
| Manhattan, dose 50 | 0.850 | 0.800 | 5.0 [−1.1, 10.6] |
| Manhattan, dose 200 | 0.870 | 0.838 | 3.2 [−2.1, 8.3] |
| Manhattan, noise-free | 0.880 | 0.827 | 5.3 [0.5, 10.8] |
| via (zero-shot) | 0.838 | 0.809 | 2.9 [−1.6, 7.6] |
| curvy (zero-shot) | 0.887 | 0.808 | 7.9 [2.6, 13.1] |

No population's point estimate exceeds 10. The CIs for Manhattan at dose 20, dose 50 and noise-free, and for curvy, reach above 10 at their upper ends.

**Where the loss comes from** (share called KILLER, all populations pooled, n = 192 per kind):

| Kind | CAD | R2 | R-oracle |
|---|---|---|---|
| bridge (want high) | 99.5% | 100% | 100% |
| open (want high) | 70.3% | **79.2%** | 81.2% |
| particle (want high) | 97.9% | **81.2%** | 96.4% |
| notch near-miss (want low) | 23.4% | **15.6%** | 37.5% |
| filament near-miss (want low) | 32.3% | **62.0%** | 49.0% |
| nuisance (want low) | 7.3% | 10.4% | 26.6% |

- **Pseudo-CAD quality:** correct topology on the central window for 92 of 96 references (95.8%). All 4 failures are the same layout (Manhattan layout 13) at every dose, so the failure is systematic, not noise.
- **Cost:** median 6 ms to build a reference.
- **Total change:** 322 of 1,152 verdicts differ between CAD and R2 modes.

## What this tells us

1. **Reference-die mode is viable on synthetic data.** It costs about 5 F1 points, half the pre-registered kill threshold. Track C Rung 3 (public SEM defect datasets) is not blocked.
2. **The loss is not segmentation.** The oracle arm, which uses the reference die's *true* wafer, loses the same 4.5 points. The cost comes from comparing against another real die, with its own LER and shift, instead of ideal CAD. A better pseudo-CAD can't remove it. More reference dies, or a median of many, might.
3. **It reshuffles errors rather than degrading uniformly.** It is better on opens and notches, and worse on particles and on filament near-misses (called killer twice as often). A dim isolated particle that appears in the target but whose location is also slightly perturbed in the reference is the likely particle mechanism. This is not yet diagnosed.
4. **The nuisance false-killer rate rises from 7.3% to 10.4%.** The oracle arm's 26.6% shows that the segmentation's smoothing actually *helps* here: a raw, LER-rough reference wafer makes a worse reference than a smoothed pseudo-CAD.

## Limitations

- Synthetic data only; the reference dies are clean by construction, but on real data a "clean" neighbour may carry its own defect. The double-reference median (R2) only partly protects against that.
- Conductor polarity is assumed (brighter = conductor). On real data it must be given.
- Val layouts 8–23 were never used for v3 selection, but they are val, not a fresh test seed.
- Thresholds were not re-selected for reference-die mode. A val-selected gate could recover some of the near-miss loss, but that would need its own pre-registration.

## Next

- **Track C Rung 3:** find a public SEM defect dataset with repeated dies or patterns and an acceptable licence. Pre-register, then run reference-die mode once.
- **Optional, synthetic:** diagnose the particle-recall loss and the layout-13 pseudo-CAD failure.
