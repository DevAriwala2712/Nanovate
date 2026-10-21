# Track C Rung 1b: three robustness changes (chosen on synthetic val) and ONE repeat on real images

**REAL DATA. The reference is FITTED, not CAD.**
- **Pre-registration:** `docs/REAL_DATA_RUNG1B_PREREG.md`, Part A at 4044b41, Part B and the frozen config at 8da0f71.
- **Run:** `python -m experiments.real_rung1b --run`, once (`results/real/rung1b/RUN_DONE`).
- **Outputs:** `results/real/rung1b/` (`tiles.csv`, `summary.csv`, `calibration.csv`, `config.json`, `val/`).

## 1. What changed (all three chosen and frozen on synthetic Manhattan val)

| # | Change | Synthetic val (1,152 patches + 288 shift pairs) |
|---|---|---|
| C1 | Calibration guard: leave-one-scribe-patch-out contrast must keep its sign and stay within 50% → else **UNVERIFIED_CALIBRATION** | triggered on 0 of 175 synthetic calibration cells → kept |
| C2 | Template clamp: T(C) clipped to the [0.5, 99.5] percentiles of the scribe-patch intensities | 0 verdicts changed; killer F1 0.865 → 0.865; shift stability 89.2% → 89.2% → kept |
| C3 | Bounded alignment: Module 3 shift > 12 px on either axis → zero shift, flagged | 0 verdicts changed; 0 rejections → kept |

**What val can and cannot say.** It shows the changes are **harmless** on the synthetic testbed. It cannot show they **help**, because the synthetic images never trigger them. The mechanisms they target (rim ringing, sign-unstable calibration, lattice jumps) were diagnosed from the Rung 1 real images. So the real images informed the design, though not any threshold. That is why a fresh, never-scored held-out set was added.

## 2. Results vs pre-registered targets

Target: KILLER on clean, overlay-free tiles ≤ 5% **and** coverage ≥ 80%. 95% CIs come from a bootstrap over images (1000 resamples, seed 0).

| Population | KILLER on clean tiles | Coverage | Met? |
|---|---|---|---|
| **fresh (held out, never scored before)**: 20 Gla_01 single-well images; 68 film-only tiles in 17 images | **0% [0, 0]** (0 / 68). With zero events, the one-sided 95% bound over images is 3/17 = 18% | 100% | **YES**, but film only (§4) |
| **orig (Rung 1's tiles, paired)**: 158 tiles in 32 images | **13.3% [5.7, 23.6]** | **73.4%** [50.5, 100] | **NO** (both conditions missed) |
| — orig / G29 (116 tiles, 29 images) | 18.1% [9.5, 28.4] (Rung 1: 70.7% [60.3, 81.9]) | 100% | — |
| — orig / G01 (42 tiles, 3 images) | 0%: all **UNVERIFIED_CALIBRATION** (Rung 1: 100% KILLER) | 0% | — |
| orig, guard off (pre-declared secondary) | 39.9% [17.2, 59.7] | 100% | NO |

**Review load (NEAR_MISS on clean tiles; reported, no target):**

| Population | NEAR_MISS | Rung 1 |
|---|---|---|
| G29 | **56.9% [45.7, 68.1]** | 25.0% |
| fresh | 8.8% [1.5, 16.2] | not scored |

**Other numbers:**
- Tiles touching the display cross (not in the target): fresh 2 of 34 KILLER (5.9%); orig 19.0% (58 tiles).
- C1 leave-one-out contrasts:
  - G29 64–75 (full fit 70): reliable.
  - FRESH 48–55 (full fit 51): reliable.
  - G01 −33, −13, **+0.8** (full fit −7.8): unreliable. The guard caught exactly the calibration Rung 1 diagnosed as broken.
- C3 rejected 8 of 158 orig alignments, all in G01 (where Module 3 had jumped by up to 138 px), and none in G29.
- Time: median 186 ms per tile; calibration plus leave-one-out about 4 x 48 ms.

**Paired with Rung 1, G29 (116 tiles), Rung 1 verdict → Rung 1b verdict:**

| Rung 1 verdict (tiles) | Now KILLER (same kind) | Now NEAR_MISS | Now NUISANCE |
|---|---|---|---|
| KILLER_BRIDGE (64) | 16 | 41 | 7 |
| KILLER_OPEN/PARTICLE (18) | 5 | 13 | 0 |
| NEAR_MISS (29) | 0 | 12 | 17 |
| NUISANCE (5) | 0 | 0 | 5 |

The C2 clamp is the only change active on G29, so it accounts for all of the G29 movement.

## 3. What this tells us

1. **The template clamp removes most fake topology on the hole arrays.** G29 false KILLER fell from 70.7% to 18.1% (same 116 tiles, paired). Rung 1's suspected mechanism is supported: an unphysical template undershoot at the rim created fake conductor rings and slivers.
2. **But it mostly turns false killers into false near-misses.** On G29, 57% of clean tiles are flagged NEAR_MISS, up from 25%. Those would go to review. The topology is now right, but the SDF energy at real rims is still high: the real holes are about 40 nm wider than the design diameter, and the real edge halo isn't modelled. The near-miss gate (e* = 2.57) fires.
3. **The guard works as a refusal, not a repair.** The G01 calibration is now flagged unreliable instead of producing 100% false killers. Coverage on orig drops to 73%, below the 80% bar, so orig fails on both counts.
4. **On fresh film (held out), the verifier is quiet:** 0 of 68 KILLER, 9% NEAR_MISS, calibration reliable.

## 4. What it does NOT tell us

- **Fresh is film only.** Each held-out field is centred on its well under the burned-in cross, so every well tile was outside the target by the pre-registered rule. Fresh therefore tests film texture, debris and charging on new images, **not** hole rims. Its 0% doesn't contradict G29's 18%.
- **No recall.** There were no defects. The clamp could hide real dim defects as well as fake ones. That needs Rung 2 (the test chip) or a synthetic defect-at-the-clamp-limit check.
- **The design leak remains for orig.** Orig's improvement is measured on the images that motivated the change. It's useful as a mechanism check, not as an unbiased estimate.
- **Reference limits remain:** a FITTED reference, design-vs-real diameter mismatch, visual "clean" labels by the assistant, FIB holes in a metal film rather than lithographic lines, and native pixels far from the 1 nm/px design point.

## 5. Next (not done)

- The near-miss gate on real rims (57% of G29) is now the dominant review cost. Candidates, each to be chosen on synthetic val only:
  - a real-CD-aware reference (Rung 2's true CAD plus a measured-CD map);
  - an edge-halo term in the simulator, so e* can be re-selected under halos.
- Rung 2 (test chip, true CAD, deliberate defects) is the first point at which recall, and therefore the clamp's cost, can be measured on real images.
