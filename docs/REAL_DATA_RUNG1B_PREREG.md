# Track C Rung 1b pre-registration: three robustness changes, chosen on synthetic val, then ONE repeat on real images

*Part A is committed before any candidate is evaluated. Part B is committed before the real repeat. The repeat uses the
same run guard as Rung 1 (committed pre-registration, run once: `results/real/rung1b/RUN_DONE`).*

**Disclosure.** All three changes target failure mechanisms diagnosed from the Rung 1 real run
(`docs/REAL_DATA_RUNG1.md` §4). The real images therefore informed the *design* of the changes, though not any threshold
value. Each constant below is a round a-priori value. To limit the leakage, Part B adds a **fresh held-out real set** that
no version of the verifier has ever scored.

## Part A: candidates and the synthetic-val selection rule (before any candidate is run)

### Candidates (new code; the v1/v2/v3 paths stay byte-identical)

| # | Change | Rule | Constant (a priori) |
|---|---|---|---|
| C1 | **Calibration guard** | Refit Module 1 leaving out each of the 3 scribe patches in turn. The calibration is UNRELIABLE if any leave-one-out contrast (μ_cond − μ_diel) has the opposite sign to the full fit, or differs from it by more than 50% of the full contrast. Every tile verified with an unreliable calibration gets the verdict **UNVERIFIED_CALIBRATION** (not KILLER, not NUISANCE). | sign agreement; relative spread ≤ 0.5 |
| C2 | **Template clamp** | The Module 1 template T(C) is clipped to [q₀.₅, q₉₉.₅], the 0.5th and 99.5th percentiles of the scribe-patch intensities. It feeds the alignment, the SDF energy and the field. | 0.5 / 99.5 percentiles |
| C3 | **Bounded alignment** | If Module 3's estimated shift exceeds 12 px on either axis, the shift is set to (0, 0) and the tile is flagged `align_rejected`. | 12 px = Exp 4 range (±8 px) + 4 px |

### Synthetic validation (Manhattan val only; no test data; seed 0)

**Patches:**
- val layouts 0–23 x 2 dies x 6 kinds;
- doses 20, 50 and 200 e⁻/px (LER 1 nm, grain 15%), plus a noise-free dose-50 copy;
- 1,152 patches in all.

**Shift-stability set:** the same 288 dose-50 patches re-rendered with a random shift of ±8 px per axis (Exp 4's range). The metric is the fraction of verdicts unchanged against the unshifted patch.

**Variants:** frozen v3 (`results/v3/calib/decisions.json`); v3 + C2; v3 + C3; v3 + C2 + C3. C1 is evaluated separately as a trigger count.

**Acceptance (each candidate on its own):**
- **C1** is kept iff it triggers on **0** synthetic calibrations. Checked over every val calibration used here and the Exp 1 grid cells (LER 0.5–2.5 nm x grain 0–30%) at doses 5–200, all on seed 0's scribe patches.
- **C2 and C3** are each kept iff, against frozen v3 on the same patches:
  1. the paired killer-F1 difference (1000-resample bootstrap) has point estimate ≥ −0.01 and a CI that includes 0 or lies above it; and
  2. shift stability is not lower by more than 1 percentage point.

  Changed verdicts are counted and reported.
- **Combination:** if both C2 and C3 pass, the frozen Rung 1b variant is v3 + C1 + C2 + C3 (keeping only the components that passed). The result is written to `results/real/rung1b/val/decisions.json` and frozen there.

No other change is allowed. If every candidate fails, the real repeat is not run, and that is reported.

## Part B: the real repeat (written after Part A's result, committed before the real run)

**Frozen variant.**
- **Primary:** frozen v3 plus the components Part A accepted, as recorded in `results/real/rung1b/val/decisions.json`.
- **Pre-declared secondary:** the same without the C1 guard. It shows what the C2/C3 changes alone do to the verdicts.

Same v3 thresholds and event rule, Window(256, 16), native pixels only. Arm B is dropped, because it stayed 100% KILLER on G01 in Rung 1 and adds nothing.

**Populations (`python -m experiments.real_rung1b --run`):**

| Population | Images / tiles | Reference | Calibration (Module 1 stand-in scribe patches) |
|---|---|---|---|
| **orig** | the Rung 1 arm C tiles exactly (G29 + G01: 174 + 42 tiles; primary = 158 clean overlay-free tiles in 32 images); paired tile by tile with Rung 1's verdicts | the Rung 1 FITTED references, unchanged | the Rung 1 calibration tiles, unchanged |
| **fresh (held out)** | the 20 Gla_01 single-well top views (doi:10.5281/zenodo.8060099, CC-BY-4.0; 1.35–3.37 nm/px, 10 kV), excluded from Rung 1 and never scored by any version. 17 evaluation images x 6 tiles = 102 tiles; primary = the **68 overlay-free tiles** (film only: each field is centred on its well, under the burned-in centre cross) | one disc of the **design** diameter (file name) at the well centre. The position comes from the dark blob nearest the image centre, because each field is centred on its well | images 0, 10, 19 of the sorted list (`z110_d_100_21`, `z50_d_220_10`, `z80_d_280_20`). For each, the 256 x 256 window centred on the well. **Disclosed:** these windows contain the centre-cross pixels (about 60 of 65,536, grey 226), because no well-containing window avoids them |

**Overlay.** For the 512 x 768 Gla_01 frames, the overlay is the 31 x 31 box at rows 241–271, cols 369–399. That is the same position as G29's R ≠ B cross, and the grayscale frames carry the cross at grey 226 inside it. Tiles touching the box are reported separately, as in Rung 1.

**Labels.** `results/real/rung1b/fresh_clean_labels.csv` holds a visual check by the assistant before any run on this set: all 20 clean (one well, no extra hole). Debris, contamination bands and smudges are nuisance. The orig labels are Rung 1's.

**Metrics and targets** (bootstrap over images, 1000 resamples, seed 0; n tiles and n images beside every number):

| Metric | Target | Met if |
|---|---|---|
| **KILLER on clean overlay-free tiles** (UNVERIFIED_CALIBRATION counts as not KILLER) | ≤ 5% | point estimate ≤ 5% **and** coverage ≥ 80% |
| **Coverage** = verified fraction of those tiles | ≥ 80% | (part of the line above) |
| KILLER among verified tiles | none | reported |
| KILLER if the guard were off (the unguarded verdict) | none | reported |
| Paired with Rung 1 (orig only): tiles whose verdict changed | none | reported |

The target is judged separately on **orig** and **fresh**. Fresh is the less contaminated test, so if the two disagree, fresh is the headline.

**What would make this not a pass even if the numbers meet the target:** none of it is decided after the run. Reporting includes both variants and every population. A low KILLER rate reached only through abstention fails the coverage condition and is reported as "abstained", not as "no false alarms".
