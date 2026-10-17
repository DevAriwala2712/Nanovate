# v2 fixes: what changed, why, and what it measured (synthetic data)

v2 is a **post-hoc revision**: every fix below was designed after the v1 test results had been seen.
To keep the second look honest:

- The pre-registered targets (AGENTS.md section 6) are unchanged.
- v1 results stay in `results/` and in the README "v1" block; v2 writes to `results/v2/` and never overwrites v1
  (`python scripts/reproduce.py --version v1` regenerates v1).
- Every new rule or threshold is chosen on **Manhattan val only**, frozen in `results/v2/calib/decisions.json`,
  then run once on test. Nothing is tuned on test or on families B/C.
- The val evidence comes from `scripts/v2_probes/` (val split only).

The v2 results table in the README is generated from `results/v2/` by `scripts/reproduce.py`; no number in
this file is a test result.

## Diagnosis of the v1 misses

| Exp | v1 miss | Root cause | Type |
|---|---|---|---|
| 1 | 46% suppression at LER sigma 1.5 nm (target 90%) | (a) spurious holes at the splice seam; (b) near-miss thresholds chosen at nominal LER, at the edge of the nuisance tail | (a) implementation bug; (b) selection protocol |
| 2 | F1 0.81-0.85 at dose >= 50 (target 0.9) | killers below the optical resolution (short opens, particles 3-4 px from a wire); also the false killers of 1(a) | testbed physics + approach |
| 4 | 94% unchanged (target 95%, CI contains it) | ~0.4 px floor from aligning to the CAD template; marginal killers flip with sub-pixel phase | inherent floor |
| 6 | 79 ms calibration (target 50 ms) | ridge fit on a 196,608 x 5 design matrix; re-blurring per registration pass | implementation |

## Fix 1 - calibration latency (Exp 6), implementation

`modules/calibration.py`: `_fit` solves the same ridge normal equations from 5 x 5 moment sums instead of
the full standardised design matrix; `_shifted_features` shifts the already blurred channels (bilinear shift
and Gaussian blur commute) instead of re-blurring a shifted CAD each pass.
Outputs are equivalent: levels, noise gain and shifts match the v1 code on 12 scribe sets (doses 5-200,
3 seeds; one noise-gain value differs in the third decimal from a flat-mask edge). Tests:
`test_moment_ridge_equals_design_matrix_ridge`, `test_shifted_features_equal_features_of_shifted_cad_in_interior`.
Scratch timing (50 runs, this CPU): 76 ms -> 41-43 ms. The Exp 6 measurement is in the README.

## Fix 2a - splice-seam holes (Exp 1, 2), implementation bug

Outside the relevance region the verifier replaces the field by the CAD (v1 "splice"). Where an image edge,
moved by LER or residual shift, does not meet that CAD fill, the fill closes a dielectric pocket into a hole
that does not exist in the wafer. On val, the long-lived false holes on nuisance samples sat on the region
boundary. `verify_topology(h1_fill="dielectric")` counts holes with dielectric outside the region, for
image and CAD reference alike; components still use the CAD fill. Tests:
`test_dielectric_h1_fill_removes_splice_seam_hole`, `test_dielectric_h1_fill_keeps_holes_inside_the_region`.

Val (Manhattan val, doses 20/50/200 pooled, n = 48 nuisance per row, v1 config otherwise; `h1fill_val.py`):

| LER sigma | false-killer v1 | false-killer v2 | bridge recall v1 / v2 |
|---|---|---|---|
| 1.0 nm | 0.08 | 0.00 | 0.94 / 0.90 |
| 1.5 nm | 0.15 | 0.06 | 0.94 / 0.92 |
| 2.0 nm | 0.27 | 0.06 | 0.92 / 0.94 |

Cost: a real bridge whose new hole touches the seam is no longer counted (2 of 48 at sigma 1.0).
The config selection grid uses the same hole rule, so tau and the field option were re-selected with it.

## Fix 2b - near-miss rule robust to roughness (Exp 1, 5), selection protocol + new statistic

v1 picked the smallest (d*, e*) with <= 5% NEAR_MISS on val nuisance at the nominal LER (1.0 nm), which put e*
at the edge of the nominal nuisance energy distribution. Nuisance max energy rises with LER while near-miss
energy does not (val, dose 50: nuisance median 1.6 at sigma 1.0 -> 2.9 at sigma 2.0; near-miss ~3.8).

Tried on val and rejected: energy normalised by an LER estimate (over-corrects: 41-56% alarm at sigma 0.5;
`nm_norm_probe.py`); masking the edge band (no effect, energy maxima are deep inside features; `nm_energy_probe.py`).

Kept:
1. **Peak-z** (`align_sdf.peak_z`): robust z-score of the largest local energy peak against the patch's other
   peaks (peak radius 20 px = nominal LER correlation length, fixed a priori). Label-free, per patch.
2. **Robust selection** (`select_config.near_miss_thresholds_robust`): (d*, e*, z*) maximise near-miss recall at
   the nominal nuisance (as v1) subject to NEAR_MISS on <= 5% of val nuisance in **every** val cell
   (LER sigma 0.5/1.0/1.5 nm x grain 0/15/30%, doses pooled).

Val simulation at a 5% worst-cell budget: v1 form (d*, e*) keeps 77% near-miss recall, adding z keeps 91%
(`nm_select_sim.py`). At LER sigma 2.0 nm roughness and near-misses overlap physically (energy AUC 0.67-0.84),
so no rule separates them there; that is outside the target range.

Disclosure: using LER up to 1.5 nm in val selection uses the target's condition range, which was decided
after seeing v1. It is val data on val layouts only.

## Fix 2c - CAD reference cache (Exp 6 regression from Fix 2a), implementation

Fix 2a added two persistence calls per patch; the first full v2 run measured Module 4 at 142 ms/patch (v1 79).
The CAD-only reference (relevance region, H0 and H1 reference diagrams) is now cached per CAD clip
(`topology._reference`). Outputs are identical (0 of 60 v2 test verdicts and reasons changed;
`test_reference_cache_returns_identical_verdicts`); Exp 6 was rerun after the cache.

## Exp 1 finding - bridges dimmed by grain (not fixed)

Exp 1 bridge recall at grain 30% is 79% at every LER (v1 and v2). On val, a dark grain on a narrow bridge drops
the conductor probability along it to 0.33-0.58, below the merge level 1 - tau = 0.65 (`grain_bridge.py`); the
missed bridges come out as NEAR_MISS, not NUISANCE. A larger tau (0.55) recovers them on val but costs open recall
(0.84 -> 0.44) and calls 42/96 filaments bridges (`grain_tau.py`), so tau stays as selected.

## Exp 2 diagnostic - verifier ceiling without shot noise (not a target)

`exp2_dose.py` writes `results/v2/exp2/ceiling.csv`: the verifier on the noise-free expected image of the
same test samples at the nominal dose. Killers it misses even noise-free are below this verifier's optical
resolution; their count per class is reported. This splits the Exp 2 gap into a noise part and a
resolution/approach part. It does not change the primary F1, whose labels still come from the wafer topology.

## Not changed

- Exp 4: no fix. The residual flips are the CAD-template alignment floor and sub-resolution killers.
- Module 2 stays timed-only (M5 result unchanged).
- Real-world data: none exists publicly with CAD and topology labels; see the fixes document's real-data
  section for options (Carinthia-S, imec line/space sets, Nebula/ARTIMAGEN independent simulators).
