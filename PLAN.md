# PLAN.md — Nanovate training-free SEM topology verifier (v4.0 Part A)

Status: **draft for approval. No implementation code written yet.**
Contract: `docs/AGENTS.md` (precedence: AGENTS.md > v4.0 Part A > protocol > v3.0).
All numbers that will appear in results are **synthetic data** and none come from the docs' scorecards.

---

## 0. What I found (exploration log)

### Docs read
| Doc | Location | Notes |
|---|---|---|
| AGENTS.md | `docs/AGENTS.md` (also inside `docs/files.zip`, identical) | Contract. `files.zip` also holds a `CLAUDE.md` whose only content is `@AGENTS.md`. |
| Plan v4.0 | `docs/Defect Detection System Plan v4.0.md` | Part A read in full; Parts B/C noted as out of scope. |
| Surgical protocol | `docs/surgical_experimental_protocol_zero_hour_universal_inspection.md` | Testbed equations, Modules 1–4 math, reference Module 2 + Module 4 code. |
| Bottlenecks | `docs/existing_semiconductor_inspection_systems_bottlenecks.md` | Background only (D2D/double-difference, MDAT, SEM SE/bloom physics). |
| v3.0 plan | `docs/(v3.0) Defect Detection System Plan.docx` | Extracted text. Code sketches: `ScribeLinePhysicsLock`, randomized SVD (torch), blind-spot denoiser, integer phase correlation + SDF, persistence + bottleneck arbiter. |
| BIOS template | **not in repo**; found at `~/Downloads/BIOS_V2_CORE_FRONTIER_Screening_Template.pptx` | Slide 4 wants a team-drawn Input→Processing→Output diagram + "what is implemented in the hack"; slide 5 wants build/risks (≥2)/impact metrics (1–2). Our outputs must supply: pipeline diagram, results table with baselines, nuisance suppression %, setup time per layer (calibration ms). |

### v4.0 vs AGENTS.md reconciliation (re-read of v4, including the docx version in ~/Downloads)
The `.md` drops the pipeline figure (`[embedded content]`). The docx (`~/Downloads/Defect Detection System Plan v4.0.docx`, `word/media/image1.png`) shows:
Inputs (SEM patch + CAD clip; 3 scribe patches) → 1 Calibrate (ridge on 5 CAD features → template + noise model) → 2 Stack (32 patches, rank-2 SVD → sparse residual; singletons denoised) → 3 Align and score (shift-correct; SDF energy boosts residuals in gaps and wire cores) → 4 Topology verifier (Δβ + bottleneck) → **Killer** (Δβ≠0) / **Near-miss** (Δβ=0, high distance + energy) / **Nuisance** (Δβ=0, low distance: dropped).

| v4.0 says | AGENTS.md says | We follow |
|---|---|---|
| Module 2 *before* Module 3 (figure order) | Align every patch to the Module-1 template **before** stacking (§5.5, bug #3) | AGENTS: M1 → M3-align → M2 → M3-SDF → M4. The README diagram will show this order and note the change. |
| "Sweep τ … pick it per layout family" (H12–18) | Label-free τ from Module 1; fallback sweep on **Manhattan val only**, never B/C | AGENTS. A per-family τ would invalidate Exp 3 (zero-shot). |
| "Module 4: already in the repo" / "the existing topology repo" | "may already exist; read it, run it" | No such code on this machine (searched Desktop/Downloads/Documents for gudhi/Betti code). Module 4 is built from the protocol §5 sketch. |
| Appendix: "Modules 1, 2 and 4 (SDF energy optional)" | Module 3 incl. SDF energy is MUST | AGENTS / Part A table: Module 3 is built. |
| Alignment "to CAD" (H18–24) | Align to the Module-1 template | Template (SEM-like appearance; same thing in CAD coordinates). |
| Preliminary repo numbers may go on today's PPT | Every number from `reproduce.py` | Any number shown is generated, labelled "preliminary, synthetic data". |

### Decisions taken on §8 questions (you said "start working"; any of these can be reversed)
Q1: `git init` inside `BIOS/` on `main` (the Desktop repo is untouched). Q2: copy AGENTS.md + `CLAUDE.md` to the root. Q3: field option (c) and the per-stack d* rule are added as *candidates* only. Q4: simulator gets documented model mismatch (σ_edge ≠ σ1, saturating bloom, anisotropic beam), with defaults fixed now, before any results. Q5: CNN gets [image, CAD] 2-channel, primary. Q6: no new deps; notebook stays thin over tested modules. Q7: conductor 4-connected.

### Existing code
**There is none.** `BIOS/` contains only `docs/`. AGENTS.md §4 says a topology prototype "may already exist", and v4.0 says "Module 4: already in the repo". Neither is true for this folder. The only existing code is the reference sketches inside the docs, so the audit in §1 covers those sketches.

### Repo / git state (needs a decision, see §8 Q1)
`BIOS/` is **not its own git repo**. The enclosing repo is `~/Desktop` (branch `master`, zero commits, with your whole Desktop untracked). Committing from here would risk sweeping in unrelated personal files. I have not touched git.

### Environment (AGENTS.md §8)
- `.venv/` created at `BIOS/.venv` with **Python 3.11.15** via `uv`. The system `python3` is 3.14, so I avoided it because wheel coverage there is a risk.
- Installed only the allowed deps:

| package | version |
|---|---|
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| scikit-image | 0.26.0 |
| gdstk | 1.0.1 |
| gudhi | 3.13.0 |
| torch | 2.14.1 |
| matplotlib | 3.11.2 |
| pandas | 3.0.6 |
| pytest | 9.1.1 |

- Hardware: Apple **M3 Pro**, 11 CPU cores, 18 GB RAM. **No CUDA GPU.** Apple **MPS is available** (torch reports `mps=True`). Plan stays CPU-first. MPS is optional, and only for the stretch items or for CNN training speed, never for timed comparisons.
- All of gudhi, gdstk and torch import cleanly.

### Smoke checks I ran (scratch scripts only, not in repo)
| Check | Result | Consequence for the plan |
|---|---|---|
| GUDHI connectivity, two pixels touching only at a corner | `CubicalComplex(top_dimensional_cells=…)` → **1 component (8-connectivity)**; `CubicalComplex(vertices=…)` → **2 components (4-connectivity)** | Labels (skimage) and PH must use the **same** convention. Proposal: conductor is 4-connected (vertex construction) and dielectric/holes are 8-connected, so a 1 nm diagonal corner touch is not a short. Pinned by a test. |
| Bug #4 reproduction: two wires plus a 0.5-brightness bridge, τ = 0.35, binary-style field | β = (2, 0), same as CAD → **bridge missed** | Confirms §7.4. Fix is in §5. |
| Phase-correlation sign (`skimage.registration.phase_cross_correlation`, upsample 20) | Applying shift (+3.4, −5.7) returns **(−3.4, +5.7)** when the test image is shifted in Fourier space. So the returned value is the correction to apply, `ndi.shift(moving, returned)`. | Sign convention confirmed and will be pinned by a test. **Pitfall:** creating the test shift with `ndi.shift(..., mode="wrap")` gives a spurious ~0.5 px error (−3.9, +6.0). `mode="grid-wrap"` gives about 0.1 px. Tests will generate shifts in Fourier space. |
| Cost of cubical PH on a 256² field | ~59 ms, single core | 1,000 patches × (1 image PH + cached reference) is about 1–2 min per configuration on one core. Multiprocessing over 5–10 workers keeps `--quick` within minutes. |
| Bottleneck distance (~770 points) | ~11 ms | Affordable for every patch. |

---

## 1. Audit of existing (reference) code: keep / fix / replace

| Item (source) | Decision | One-line reason |
|---|---|---|
| `compute_robust_betti_numbers` (protocol §5) | **Fix** | Core idea is right. Must switch to the vertex construction (4-connectivity), pass a 2-D array instead of `flatten(order='F')` (that transposes, which matters once we localise pairs), and take τ from calibration rather than a fixed 0.35. |
| `verify_topological_defect` (protocol §5) | **Replace** | Percentile normalisation plus fixed τ (bug #4), no boundary margin (bug #5), no near-miss branch, no reason payload. |
| `cad_hashed_rpca_decomposition` (protocol §5) | **Fix** | Uses a full SVD, not a randomized one, and runs on unaligned stacks (bug #3). The MAD soft-threshold logic is reusable. Output must be L+S per patch, not S. |
| `litho_stochastic_rpca_gpu` (v3.0) | **Fix → numpy** | Randomized subspace iteration with a seeded sketch is the right algorithm. The FNO-PVB weighting is out of scope (DO NOT build), so it gets removed. Port to numpy (CPU-first). A torch/MPS version is stretch. |
| `ScribeLinePhysicsLock` (v3.0) | **Fix** | The 5-feature ridge is correct as specified. `anscombe_stabilize` is wrong: it adds raw residual variance inside the sqrt, which is not the generalised Anscombe transform, and the residual variance mixes model error with noise. Needs per-level noise fit Var(I) = α·μ + σ²_read, then GAT. |
| `align_and_score_sdf_tension` (v3.0) | **Fix** | Integer-only, untested sign (bug #2). It aligns a residual that was computed before alignment (bug #3). `exp(γ·|φ|)` with γ = 0.18 explodes in large dielectric areas (via family, |φ| up to ~100 px → e¹⁸), so |φ| must be capped. SDF sign convention (φ > 0 in dielectric) is fine. |
| `SelfSupervisedBlindSpotDenoiser` (v3.0) | **Stretch, fix** | `nn.Dropout2d` on 1 channel blanks the whole image (bug #1). Will be replaced with per-pixel masking. |
| `FewShotTopologicalArbiter` (v3.0) | **Mine for parts, replace** | Reuse the bottleneck call and verdict ordering. Drop Δβ₂ (DO NOT build) and the ADC/active-learning head (stretch at most). Its d* = 0.18 and e* = 0.45 are doc numbers and will not be used. Its bottleneck is computed against the *binary CAD* diagram, whose points sit at (0,1). That distance is dominated by global contrast mismatch, even for clean patches (see §4.3). |
| `compute_persistence_image_and_asph` (v3.0) | **Not building (stretch only)** | Persistence-image classifier is stretch. "ASPH" framing is parked in Part C. |
| Phase correlation (bottlenecks doc formula) | **Replace** | Use `skimage.registration.phase_cross_correlation(upsample_factor≥20)` as AGENTS.md §5.4 says. |

---

## 2. Proposed repo layout

Nothing exists, so I'll adopt AGENTS.md §4's layout as-is, with the small additions marked ➕:

```
BIOS/
  docs/                          # untouched source docs
  pyproject.toml          ➕     # `pip install -e .` so experiments/tests import nanovate
  .gitignore              ➕     # .venv/, __pycache__/, results/**/cache, *.npz caches
  src/nanovate/
    config.py                    # dataclass configs (TestbedCfg, SEMCfg, ModuleCfgs, ExpCfg); to/from JSON
    rng.py                ➕     # one seed → SeedSequence.spawn per sample/worker; torch seeding
    topo_utils.py         ➕     # shared: Betti-of-binary (skimage, matching connectivity), PH wrapper, central-window helpers
    testbed/  layouts.py perturb.py sem.py labels.py dataset.py scribe.py➕
    modules/  calibration.py lowrank.py align_sdf.py topology.py fields.py➕
    baselines/ die_to_die.py cnn.py
    metrics.py            ➕     # F1/recall/suppression + bootstrap CIs (shared by all experiments)
    pipeline.py                  # verdict for one patch / one K-stack, returns verdict + reason dict
  experiments/  common.py➕  exp1_nuisance.py … exp6_latency.py
  scripts/reproduce.py           # --quick / full; runs exp1–6, writes results/, regenerates README table
  tests/                          # pytest, one+ per module and per defect class
  results/expN/                   # CSV + PNG + config.json (+ frozen_thresholds.json in results/calib/)
  notebooks/demo.ipynb
  PLAN.md  PROGRESS.md  README.md
```
Justification for the additions:
- `fields.py` keeps field construction options (§4.1) separate from PH, so they can be swapped and compared.
- `topo_utils.py` puts the connectivity convention in one place, because labels and verdicts must agree.
- `metrics.py` and `experiments/common.py` give all six experiments and both baselines one shared scoring and CI code path.
- `scribe.py` separates scribe-patch generation from in-die tiles.
- `pyproject.toml` avoids `sys.path` hacks.

No other restructuring. Proposed but needing your OK (§8): copy `AGENTS.md` and a `CLAUDE.md` (`@AGENTS.md`) to the repo root so future sessions auto-load the contract.

---

## 3. Milestones (MUST order of AGENTS.md §3)

Time estimates are rough wall-clock for me, including tests and a first run. Each milestone ends with: `pytest -q` green → `PROGRESS.md` updated with measured numbers, surprises and next steps → one small commit.

### M1 — Scaffold + synthetic SEM testbed (~6–8 h)
**Deliverable:**
- `pyproject.toml`, `config.py`, `rng.py`.
- `testbed/` package:
  - **Layouts** (gdstk polygons): A Manhattan (14 nm half-pitch lines, with jogs/line ends so opens and bridges have variety), B curvy spline wires (`gdstk.FlexPath` along cubic splines, min width and space ≥ 14 nm enforced by checking the rasterised SDF), C via/pillar arrays.
  - **Rasterisation** with `skimage.draw.polygon` at 1 nm/px. A 4× supersampled antialiased coverage map is kept for the SEM model only. W is binary.
  - **Dies:** each layout = one die, rendered K = 32 times with independent LER, grain, defect, noise, shift and charging.
  - **LER:** OU process (σ, ξ = 20 nm) sampled along arc length of each polygon contour, densified to ≤ 1 nm vertex spacing, then vertices displaced along normals.
  - **Grain:** Voronoi grains on conductor, contrast U(−α, α), α ≤ 30%.
  - **Defects:** bridge capsule across gaps < 25 nm (r 3–10 nm); open (void cutting fully across a trace); particle (isolated blob in dielectric, not touching any conductor); near-miss notch (constriction that leaves ≥ 1 px of 4-connected path); near-miss filament (capsule from one wire stopping 1–3 nm short of the neighbour).
  - **SEM model:** G_σbeam * (a0 + a1·W + a2·|∇(G_σedge*W)|) + low-order polynomial charging, then Poisson(λ·Ĩ) + N(0, σ_read²), plus a random sub-pixel/pixel shift per patch.
  - **Labels** are computed from topology of W vs CAD on the central window, never from intent. Resample on mismatch and count resamples per (class, σ).
  - **Saved per sample:** `(CAD, W, image, mask, label, params)`.
  - **Scribe patches:** 3 large isolated features per layer (pad, wide line, cross), same forward model.
  - **Splits:** separate layout seeds per split (train/val/test), so no layout leaks between splits.

**Tests:**
- Same seed → bit-identical arrays.
- Every saved sample's label equals recomputed topology.
- Nuisance-only samples have Δβ(W, CAD) = 0.
- Each defect class produces its intended Δβ (bridge β0−1 or β1+1; open β0+1; particle β0+1; near-miss Δβ = 0).
- Filament gap measured in W is 1–3 px.
- Units: image mean ≈ λ·Ĩ (e⁻/px).
- Resample counter exists and is reported.

**First measurement:** resample rate vs LER σ. Bug #6 says it will be non-zero near σ = 2.5 nm on 7 nm half-width wires.

### M2 — Module 4: topology verdict + bottleneck near-miss rule (~5–6 h)
**Deliverable:**
- `modules/topology.py`:
  - Cubical PH (vertex construction) on a conductor-probability field.
  - Betti from intervals with lifetime ≥ τ.
  - Central-window comparison with margin (§5, bug 5).
  - Bottleneck distances per dimension against the reference diagram.
  - Verdict ladder exactly as AGENTS.md §5.2.
  - `reason` dict: β_ref, β_img, Δβ, bottleneck (H0, H1), max SDF energy (filled once M4 lands; until then NEAR_MISS is disabled and reported as such), field option, τ, and the singleton flag.
- `modules/fields.py` with field options (a) and (b) from §4.1. (b) temporarily uses oracle levels from the simulator config until M3 lands. This is marked in outputs as `levels=oracle` and never used for reported results.

**Tests:**
- One per defect class on noiseless images: bridge → KILLER_BRIDGE, open → KILLER_OPEN/PARTICLE, particle → KILLER_OPEN/PARTICLE.
- LER-only → NUISANCE.
- Dim bridge (0.5 contrast) detected under the calibrated-τ rule (bug #4 regression).
- Diagonal corner touch is not a bridge.
- Wire crossing the window edge gives no verdict flip.

**Early risk probe:** false-killer rate on clean Manhattan val vs dose for options (a)/(b). This is the first look at the bloom fake-bridge risk.

### M3 — Module 1: scribe ridge calibration + generalised Anscombe (~3 h)
**Deliverable:**
- `modules/calibration.py`:
  - Features [1, G_σ1*C, G_σ2*C, |∇(G_σ1*C)|, ∇²(G_σ1*C)].
  - Closed-form ridge on the 3 scribe patches.
  - Predicted template T̂(C) and predicted conductor/dielectric levels.
  - Noise model Var = α·μ + σ²_read fitted from binned residuals → GAT.
- Optional per-patch low-order ramp removal (§4.1). Off by default; measured.
- Replaces the oracle levels in M2.

**Tests:**
- Recovers known weights on data generated exactly in the feature span.
- Contrast inversion (a1 < 0) handled.
- GAT makes variance ≈ 1 across doses (within tolerance).
- Calibration timing is recorded (Exp 6 hook).

**Measurement:** relative error ‖T̂ − E[I]‖/‖E[I]‖ on held-out in-die patches per family. This is the scribe-to-die transfer number, reported as measured.

### M4 — Module 3: sub-pixel alignment + SDF tension energy (~3 h)
**Deliverable:**
- `modules/align_sdf.py`:
  - `phase_cross_correlation(upsample_factor=20)` of the patch against T̂ (not the binary CAD, whose appearance differs from SEM), with optional Hann window.
  - Apply the shift with `ndi.shift`.
  - SDF φ from CAD via EDT (φ > 0 in dielectric, in nm).
  - Energy |S|·exp(γ·min(|φ|, φ_cap)).
  - No Langevin step.
- Wire the SDF energy into the NEAR_MISS branch.

**Tests:**
- Known Fourier shifts in ±8 px with sub-pixel parts recovered to < 0.1 px at high dose, and < 0.25 px at λ = 5 (tolerance to confirm). Both signs checked.
- Round-trip: register(shift(x)) ≈ x.
- Shifted sample → same verdict.
- SDF sign and units.
- Energy finite on the via family (cap works).

### M5 — Module 2: aligned K = 32 stack + rank-2 randomized SVD + MAD (~3–4 h)
**Deliverable:**
- `modules/lowrank.py`:
  - Align every die to T̂ first (bug #3), then stack D×K.
  - Seeded randomized subspace iteration (rank 2, oversample 2, 2 power iterations).
  - Per-pixel MAD soft threshold.
  - Per-patch field input = L_i + S_i.
- Singleton fallback: median → Gaussian denoise, flagged `singleton=True` in every output row.

**Tests:**
- Low-rank residual energy for an aligned stack is far below that of the unaligned stack.
- Injected defect appears in S of its own die only.
- Same seed → identical L/S.
- Singleton path flagged.
- Randomized SVD matches full SVD on rank-2 part within tolerance.

### M6 — Baselines (~4–5 h)
**Deliverable:**
- `baselines/die_to_die.py`:
  - |target − aligned reference die| (same Module-3 aligner, so alignment isn't a confound).
  - Pixel threshold and minimum connected area chosen on Manhattan val, then frozen.
  - Also a double-difference variant (two neighbours, same-sign residual, as in the bottlenecks doc). Both are reported, so the baseline isn't a strawman.
- `baselines/cnn.py`:
  - Small ResNet-style net (~4 residual blocks, global pool) on 256×256 patches.
  - Trained on Manhattan train only, CPU, seeded.
  - Decision threshold on Manhattan val.
  - Inputs per §8 Q5.

**Tests:**
- D2D flags an obvious bridge and does not flag an identical die.
- Threshold selection uses only val.
- CNN forward shape and determinism (two seeded runs → identical first-batch loss).
- CNN overfits a tiny batch (sanity check).

### M7 — Experiments 1–6 (~6–8 h)
**Deliverable:**
- `experiments/exp1..exp6.py`, each writing `results/expN/{*.csv,*.png,config.json}` via `experiments/common.py`.
- Each reports mean + 95% percentile bootstrap CI (1,000 resamples; 200 under `--quick`) with n beside every number.
- Pre-registered metric definitions, fixed in code before the first test-set run:
  - **Positive class** = killer (bridge / open / particle). F1 is killer vs not-killer.
  - **Nuisance suppression %** = fraction of nuisance-only samples whose verdict is NUISANCE (NEAR_MISS on a nuisance sample counts as a false alarm). "Not KILLER" rate is reported as a secondary.
  - **Exp 1:** σ ∈ {0.5, 1.0, 1.5, 2.0, 2.5} × grain ∈ {0, 10, 20, 30}%. Bridge recall from bridge samples in the same cells.
  - **Exp 2:** λ ∈ {5, 10, 20, 35, 50, 100, 200}. Report the lowest λ with F1 ≥ 0.9, or "none".
  - **Exp 3:** everything calibrated/frozen on A val, then tested on A, B, C test. Drop = F1(A) − F1(B or C), in points.
  - **Exp 4:** the same W re-acquired with shifts U(−8, 8) px plus a small smooth warp (≤ 1 px amplitude). % verdicts unchanged vs zero shift, with vs without Module 3.
  - **Exp 5:** near-miss recall at a fixed nuisance false-alarm rate, pre-registered as 5% and set on A val. Compared with the Δβ rule alone.
  - **Exp 6:** calibration ms (median of 100 runs); ms/patch per module over 1,000 patches; CNN latency at batch 1 and batch 64. Same CPU, single-thread and all-thread both reported.
- `--quick` runs everything with small n in a few minutes.

**Tests:**
- Bootstrap CI code on known distributions.
- Smoke test that every experiment runs end-to-end under `--quick` with tiny n (≤ 1 min each).
- Reproducibility: two `--quick` runs → identical non-timing CSVs.

### M8 — Packaging (~3–4 h)
**Deliverable:**
- `scripts/reproduce.py [--quick]` regenerates every CSV and plot and rewrites the README results table between markers, from the CSVs only.
- README:
  - Thesis paragraph.
  - ASCII pipeline diagram.
  - How to run.
  - Generated table with baselines, CIs and n, labelled "synthetic data".
  - Limitations: sim-to-real, τ sensitivity, singletons, scribe-to-die transfer, patch boundaries.
- `notebooks/demo.ipynb`: one tile through every stage, showing each verdict and its reason.
- PROGRESS.md notes for every missed target.

**Tests:**
- README table generator is idempotent.
- Notebook executes headless (`jupyter` isn't an allowed dep, so this needs your OK; otherwise the notebook logic lives in a module and is tested there).

**Total:** roughly 33–41 h of build. That matches the v4.0 48-hour plan with buffer for the stretch items. Cut order if late (AGENTS.md §3): Module 2 → Module 1 → CNN. Never cut die-to-die or the dose sweep.

---

## 4. Open design choices (AGENTS.md §5.2) and how each will be decided

All decisions are made on **Manhattan validation only**, recorded in `results/calib/decisions.json` with the val metrics that justified them, then frozen. Code guard: any function that selects a threshold refuses to run on a `test` split or on families B/C.

### 4.1 Field construction
Candidates (all on the aligned, central-window patch, with mild Gaussian pre-smoothing σ_s; σ_s is fixed in config, or derived from the Module-1 noise estimate if that proves better on val):
- **(a) Percentile normalisation** (reference): clip to [p2, p98] → [0, 1]. This is the baseline field, kept to show bug #4.
- **(b) Calibrated level mapping:** P = clip((I − μ_diel)/(μ_cond − μ_diel), 0, 1), using the Module-1 predicted conductor and dielectric levels. This handles contrast inversion automatically, but bloom still brightens narrow gaps.
- **(c) ➕ (needs your OK, §8 Q3) Template-anchored residual field:** P = clip(Ĉ + (I − T̂)/Δ, 0, 1), where T̂ is the Module-1 predicted image, Ĉ is the smoothed CAD and Δ is the predicted contrast. T̂ already contains bloom, so bloom cancels and only unexplained intensity moves the field.
  - Risk: it injects a CAD prior, so faint real defects are pulled toward CAD.
  - It must therefore be judged on killer recall, not just suppression.

**Selection criterion (pre-registered):** maximise killer F1 on A-val at λ ∈ {20, 50, 200}, subject to nuisance suppression ≥ what option (a) achieves. Tie-break on fewer fake bridges in narrow-gap clean patches. Report all options in an ablation CSV whatever wins.

### 4.2 τ rule (and d*, e*)
- **Preferred, label-free:** in options (b)/(c) the field is in units of predicted conductor–dielectric contrast. So τ = κ·1 with κ = 0.5 (a connection counts once it is more than half-way to conductor), fixed a priori from Module 1, with no labels.
- **Check:** a τ sweep κ ∈ [0.1, 0.9] on A-val is plotted to show sensitivity (that is itself a README limitation). If κ = 0.5 is clearly off the val optimum, I fall back to the val-selected κ and record that in PROGRESS.md.
- **d\* (bottleneck), label-free option ➕:** within a K = 32 stack most dies are clean. The per-layout null distribution of bottleneck distances between each die's field and the reference gives d* = median + k·MAD (k fixed a priori, e.g. 3). Singletons use a d* from scribe-patch nulls instead.
- **d\* fallback:** sweep on A-val to hit the pre-registered 5% nuisance false-alarm rate. Same for e*.
- The doc values (τ = 0.35, d* = 0.18, e* = 0.45) are never used.

### 4.3 Reference topology
- **(i) Binary CAD:** β_ref from the CAD raster, exact integers.
- **(ii) Module-1 template:** T̂(C) passed through the *same* field construction and τ as the image.

Both are recorded per sample as required. Expectation, to be tested rather than assumed: for Δβ, (i) and (ii) agree unless bloom makes T̂ show a fake bridge. In that case (ii) cancels the fake bridge but also hides a real bridge at that site, and this must be counted. For the **bottleneck** distance, (ii) is likely necessary: against (i), every clean patch has distance ≈ 1 − contrast_fraction (CAD points sit at (0,1)), which swamps the near-miss signal.

**Choice criterion:** killer F1 for Δβ, and near-miss recall at 5% FAR for the bottleneck rule, on A-val. The Δβ reference and the bottleneck reference may differ. That is allowed and will be documented.

---

## 5. How each known bug (AGENTS.md §7) is addressed

| # | Bug | Fix | Test that pins it |
|---|---|---|---|
| 1 | `Dropout2d` on a 1-channel input zeros the whole image (stretch) | Per-pixel Bernoulli masking (N2V style: masked pixels replaced by a random neighbour). Loss only on masked pixels. | Masked fraction ≈ p and never 100%. ∂out[c]/∂in[c] = 0 (blind spot holds through the dilated layers). |
| 2 | Phase correlation integer-only, sign untested | `phase_cross_correlation(upsample_factor ≥ 20)` against T̂. Sign convention: returned value is the correction for `ndi.shift(moving, ·)`, confirmed in a smoke check. | Fourier-shifted inputs over a grid of ±8 px with fractional parts, both signs, error < 0.1 px. Round-trip identity. Noise-robustness at λ = 5. Avoid `mode='wrap'` when generating test shifts (measured ~0.5 px artefact). |
| 3 | Module 2 stacks before alignment | Pipeline order is fixed in `pipeline.py`: M1 template → M3 align every die → stack → randomized SVD → MAD → field = L_i + S_i → M4. | Aligned vs unaligned residual energy test. Defect isolated to its own die. Test asserts `lowrank()` refuses stacks without an `aligned=True` provenance flag. |
| 4 | Percentile normalisation + τ = 0.35 misses dim bridges and fakes bloom bridges | Field options (b)/(c) in contrast units + τ from Module 1 (§4.1–4.2). Option (a) is kept only as an ablation. | Regression: wire–wire bridge at 0.5 contrast → KILLER_BRIDGE under the calibrated rule (it fails under (a)+0.35, also asserted, to document the bug). Narrow-gap clean patch with strong bloom → not KILLER under the chosen field. |
| 5 | Topology compared without a boundary margin | CAD and image cropped to the **same central window** after alignment (margin m ≥ 8 px shift + blur support, ~16 px). Defects are injected only ≥ m + r from the tile edge. Labels use the identical window. Fallback if crop slivers cause flips: pad the margin band with the T̂-derived field so topology changes can only originate centrally. | Wire crossing the window boundary at a shallow angle gives no flip. Shifted clean sample gives the same verdict. Label window equals verdict window (asserted). |
| 6 | Nuisance injection can change topology by chance | Label from Δβ(W, CAD) on the central window with the shared connectivity convention (conductor 4-conn, holes 8-conn), never from intent. Mismatch → resample with a per-(class, σ, family) counter written to `results/testbed/resamples.csv`. Survivor bias at high σ is reported, not hidden. | Every sample re-verified at load. Property test over many seeds: nuisance ⇒ Δβ = 0, killer ⇒ Δβ ≠ 0, near-miss ⇒ Δβ = 0. Resample counter > 0 at σ = 2.5 nm is asserted to be reported, not suppressed. |

**Additional issues I found in the reference code (beyond §7), all fixed in the milestones above:**
- The v3 Anscombe formula is not the GAT.
- SDF energy overflows in large dielectric regions.
- Bottleneck distance is computed against the binary-CAD diagram.
- `flatten(order='F')` transposes the field.
- Top-cell cubical complex uses 8-connectivity, which disagrees with 4-connected labels.
- Timing columns make CSVs non-deterministic. These go in their own `exp6/latency.csv`, excluded from the identical-CSV check, which is stated in the README.

---

## 6. Top risks and the early measurement that detects each

| # | Risk | Early signal (milestone) | Mitigation if it fires |
|---|---|---|---|
| R1 | **Circular testbed.** Module 1's 5 features (Gaussian blurs, gradient magnitude, Laplacian) span my own SEM model almost exactly, so calibration looks perfect by construction. | Rel. error of T̂ on in-die patches ≈ 0 in M3 is a red flag, not a win. | Introduce deliberate, documented model mismatch in the simulator: σ_edge ≠ σ1, non-linear bloom (sec θ-like saturation), scan-direction PSF anisotropy, scribe-vs-die charging difference. **Needs your OK (§8 Q4).** |
| R2 | **Bloom fake bridges** in narrow gaps | M2 probe: false-killer rate on clean A-val vs dose, options (a)/(b) | Option (c) / reference (ii); report the residual rate. |
| R3 | **Near-miss vs bridge indistinguishable.** A 1–3 nm filament gap is smaller than the beam blur, so the image likely reads as connected and becomes KILLER_BRIDGE. | Exp-5 dry run on A-val right after M4: near-miss→KILLER confusion rate by gap size | Report honestly as a resolution limit (gap vs σ_beam plot). Do not tune it away. |
| R4 | **Low-dose spurious topology.** Noise blobs become particles or opens at λ ≤ 20. | M2/M3: false-killer rate vs λ with and without GAT and smoothing | σ_s from the Module-1 noise model; Module 2 denoising. Report the cliff. |
| R5 | **Survivor bias at high LER σ.** Resampling removes exactly the hard nuisance cases. | M1 resample rate per σ | Report rate next to Exp-1 numbers. Also report "topology-changing LER" separately as killers by label. |
| R6 | **Scribe-to-die transfer** | M3 rel. error per family + charging ramp residual | Optional per-patch ramp removal (measured), plus a limitation in the README. |
| R7 | **Boundary flips** | M2 boundary tests + verdict-flip rate across crop offsets on clean val | Margin-padding fallback (§5 #5). |
| R8 | **Threshold leakage** onto test or B/C | Code guard + `decisions.json` written before any test run (timestamped) | Hard error. |
| R9 | **Runtime** blows the `--quick` budget | M2: ms/PH measured (~59 ms now); M7 quick timing | Multiprocessing with spawned seeds; reference diagrams cached per layout. |
| R10 | **Determinism across processes/torch** | M7 two-run CSV diff test | Per-sample SeedSequence; CPU torch with `use_deterministic_algorithms(True)` and a fixed thread count for training. |
| R11 | **CNN baseline unfair** in either direction | M6: CNN in-domain A-val F1 sanity check (it should be strong in-domain; if weak, it is under-trained) | Fixed training budget stated in config. Same patch size and hardware. |

---

## 7. Determinism, splits and reporting conventions
- One `seed` in the top-level config drives everything. Sample *i* of split *s* in family *f* gets its own `SeedSequence(seed).spawn` child, so results don't depend on worker count.
- Split sizes and per-cell sample counts go in config. Quick vs full defaults are set in `ExpCfg` and saved to every `config.json`.
- Everything except timing is bit-identical across runs.

## 8. Decisions I need from you before M1
1. **Git:** the enclosing repo is `~/Desktop` (no commits, whole Desktop untracked). OK to `git init` a separate repo in `BIOS/` on branch `main`, with `.venv/` and `results/` caches git-ignored?
2. **Contract files at the root:** OK to copy `docs/AGENTS.md` and a one-line `CLAUDE.md` (`@AGENTS.md`) into `BIOS/`? `docs/` stays untouched either way.
3. **Extra design candidates:** OK to add field option (c) (template-anchored residual) and the label-free per-stack d* rule as candidates alongside the specified ones? Both are compared, not assumed.
4. **Simulator mismatch (R1):** OK to deliberately make the SEM forward model *not* lie in Module 1's feature span (documented knobs, defaults chosen before any results)? Without it, Module 1 numbers are circular.
5. **CNN inputs:** my default is a 2-channel [image, CAD raster], matching design-guided industry CNNs, with shift augmentation. Image-only is the alternative. Which do you want? (Or both, with one reported as primary.)
6. **Notebook execution test:** `jupyter`/`nbconvert` is not in the allowed deps. Add `nbconvert` (dev only) to test the notebook headlessly, or keep the notebook thin over a tested module?
7. **Connectivity convention:** conductor 4-connected (a corner-only touch is not a short), dielectric 8-connected. OK?
