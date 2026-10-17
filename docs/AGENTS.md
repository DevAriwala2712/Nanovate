# AGENTS.md — Nanovate: Training-Free SEM Topology Defect Verifier (Plan v4.0 prototype)

## 1. Mission (read this first)

Build and measure ONE thing: a training-free SEM defect verifier that calls a defect a
**killer only when circuit topology differs from CAD**, calibrated per layer from three
scribe-line patches, evaluated on a synthetic SEM testbed against honest baselines.

This is a hackathon prototype for a BIOS hack submission. The deliverable is a repo where
one command regenerates every number in the results table, plus a demo notebook.
Correctness and honest measurement beat breadth. A missed target reported honestly is a
success; an unexplained miracle number is a failure.

## 2. Source documents and how to treat them

The repo contains (find them with `ls`/`find`; do not edit them):

| Doc | Role |
|---|---|
| Defect Detection System Plan **v4.0** | **Authoritative scope**: Part A = what to build. Parts B/C = do NOT build. |
| `surgical_experimental_protocol_zero_hour_universal_inspection.md` | Math + reference code for the testbed and Modules 1–4. Design source, not results. |
| `existing_semiconductor_inspection_systems_bottlenecks.md` | Background on the problem. No code to build from it. |
| `(v3.0) Defect Detection System Plan.docx` | Superseded. Use only for the code sketches it contains. |
| `BIOS_V2_CORE_FRONTIER_Screening_Template.pptx` | Context for what the outputs must support (slides 4–5). |

Rules for the docs:
- **Every number in the docs is a hypothesis, not a target to reproduce.** The scorecards in
  the protocol and v3.0 (F1 0.97, 98.6% suppression, 310 ms, 270,000×, etc.) were never
  measured. Never hardcode, copy, or tune toward them.
- The reference code in the docs is a sketch with known bugs (Section 7). Treat it as a
  starting point to verify, not as correct.
- If the docs conflict, the order of precedence is: this file > v4.0 Part A > protocol > v3.0.

## 3. Scope

### MUST build (in this order)
1. **Synthetic SEM testbed** (SEM mode only).
2. **Module 4**: persistent-homology Δβ verdict + bottleneck-distance near-miss rule.
3. **Module 1**: scribe-line ridge calibration + Anscombe stabilisation.
4. **Module 3**: sub-pixel phase-correlation alignment + CAD SDF tension energy.
5. **Module 2**: same-layout patch stack (K = 32) + rank-2 randomized SVD + MAD threshold;
   singleton fallback (median/Gaussian denoise).
6. **Baselines**: die-to-die difference + threshold; small CNN trained on Manhattan only.
7. **Experiments 1–6** (Section 6), CSV + plots, README results table, demo notebook,
   one reproduce command.

If time runs short, cut in this order: Module 2, then Module 1, then the CNN baseline.
**Never cut** the die-to-die baseline or the dose sweep.

### STRETCH (only after MUST is done and green)
Blind-spot self-supervised denoiser (fix the Dropout2d bug first) → persistence-image
few-shot classifier over synthetic topology classes only → GPU randomized SVD.

### DO NOT build
FNO/DOINN/LithoBench surrogates, PVB-guided thresholds, optical BBP mode, Discrete Morse /
custom CUDA homology, BSE/voltage-contrast fusion, 3D X-ray / Gaussian splatting, Δβ₂,
ASPH-GNN, Causal-KAN, Langevin curvature, ROI/market code. If a task seems to need one of
these, stop and ask.

## 4. Existing repo

A basic topology-level prototype may already exist. **Before writing code**, read it,
run it, and record in `PLAN.md` what works, what's broken, and what you'll keep. Extend
rather than rewrite; restructure only if the layout below is clearly better and you say why.

Target layout (adapt to what exists):
```
src/nanovate/
  testbed/   layouts.py  perturb.py  sem.py  labels.py  dataset.py
  modules/   calibration.py  lowrank.py  align_sdf.py  topology.py
  baselines/ die_to_die.py  cnn.py
  pipeline.py            # end-to-end verdict for one patch / one stack
  config.py              # dataclass configs; every run's config is saved with its results
experiments/ exp1_nuisance.py … exp6_latency.py
scripts/reproduce.py     # regenerates every CSV, plot, and the README table
tests/                   # pytest
results/                 # generated; one subfolder per experiment with config.json
notebooks/demo.ipynb
PLAN.md  PROGRESS.md  README.md
```

## 5. Technical specification

### 5.1 Testbed (SEM mode only)
- Tiles: 256 × 256 px at 1 nm/px. Three layout families:
  **A Manhattan** lines (14 nm half-pitch), **B curvy** spline wires, **C via/pillar** arrays.
  Use `gdstk` for geometry; rasterise with `skimage.draw.polygon` or equivalent (gdstk
  does not rasterise).
- Die repetition: to support Module 2 and die-to-die, generate each layout tile as a
  "die" repeated K times, each copy with **independent** LER, grain, noise, shift and
  (sometimes) a defect.
- Nuisance: Ornstein-Uhlenbeck LER along edge normals (σ 0.5–2.5 nm, ξ = 20 nm);
  Voronoi grain contrast up to 30% on conductors.
- Defects: bridge capsule (r 3–10 nm) across gaps < 25 nm; open (void across a trace);
  particle (isolated conductive blob in dielectric); near-miss notch (wire constriction
  that does not sever) and near-miss filament (stops 1–3 nm short of the neighbour).
- SEM forward model: blur + edge bloom (|∇(G*W)|) + low-order charging ramp + Poisson dose
  λ ∈ [5, 200] e⁻/px + Gaussian read noise. Random sub-pixel/pixel shift per patch.
- **Labels come from exact topology of the ground-truth binary wafer W**, never from the
  injection intent. Compute connected components / holes of W vs CAD (same window,
  same margin). If a "nuisance" or "near-miss" sample changes topology, or a "killer"
  sample doesn't, resample it and count resamples. Save `(CAD, W, image, mask, label,
  params)` per sample.
- Scribe patches: 3 large, isolated features per layer, rendered with the same forward
  model, used only by Module 1.

### 5.2 Module 4 — topology verdict (core)
- Cubical persistence (GUDHI) on a conductor-probability field; Betti numbers from
  intervals with lifetime ≥ τ.
- Verdicts: Δβ₀<0 or Δβ₁>0 → KILLER_BRIDGE; Δβ₀>0 or Δβ₁<0 → KILLER_OPEN/PARTICLE;
  Δβ = 0 and bottleneck distance ≥ d* and SDF energy ≥ e* → NEAR_MISS; else NUISANCE.
  Every verdict returns its reason (Δβ, bottleneck distances, max SDF energy).
- **Field construction is a design decision to test, not assume.** The reference code
  percentile-normalises the image; SEM edge bloom can make narrow gaps look bright and
  fake a bridge. Implement and compare on Manhattan validation: (a) percentile
  normalisation, (b) mapping intensity to conductor probability using the Module-1
  predicted conductor/dielectric levels. Also record the reference topology from both
  the binary CAD and the Module-1 predicted template. Pick on Manhattan val; document why.
- **Boundary handling:** wires connect outside the crop. Compute topology for CAD and
  image on the same window with a margin, and compare on the central region.
- **Threshold τ (and d*, e*):** preferred is a label-free rule derived from Module-1
  calibration (e.g. a fraction of predicted conductor–dielectric contrast). Fallback:
  sweep on **Manhattan validation only**, then freeze. Never tune on test or on
  families B/C.

### 5.3 Module 1 — scribe-line calibration
Features [1, G_σ1*C, G_σ2*C, |∇(G_σ1*C)|, ∇²(G_σ1*C)]; ridge regression in closed form on
3 scribe patches; residual variance → generalised Anscombe. Report relative prediction
error on held-out in-die patches (this is the scribe-to-die transfer risk — measure it).

### 5.4 Module 3 — alignment + SDF energy
Sub-pixel phase correlation (e.g. `skimage.registration.phase_cross_correlation` with
`upsample_factor ≥ 20`); the reference code is integer-only. Write a test that recovers a
known shift and checks the sign convention. SDF φ from CAD; energy |S|·exp(γ|φ|).
No Langevin step.

### 5.5 Module 2 — low-rank stack
**Align every patch to the Module-1 template before stacking** (the protocol orders
alignment after RPCA; misaligned stacks are not low-rank). Rank-2 randomized SVD with a
fixed seed, per-pixel MAD soft threshold. The field passed to Module 4 is the patch's
denoised estimate (its low-rank part plus its sparse part), not S alone. Singletons
(no K peers) fall back to median/Gaussian denoise and are flagged as such in outputs.

### 5.6 Baselines (must be fair)
- **Die-to-die:** |target − aligned reference die| with a threshold chosen on Manhattan
  val; patch flagged if any connected region exceeds a minimum area. Same patches,
  same labels.
- **CNN:** small ResNet-style classifier, trained on Manhattan train split only, same
  patch size as the verifier, threshold chosen on Manhattan val. Time it on the same
  hardware and batch conditions as the verifier.

## 6. Experiments (pre-registered; targets are hypotheses)

| # | Experiment | Sweep | Metric | Stated target | Compared against |
|---|---|---|---|---|---|
| 1 | Nuisance stress | LER σ 0.5–2.5 nm; grain 0–30% | nuisance suppression %; bridge recall | ≥ 90% / ≥ 90% at σ ≤ 1.5 nm | die-to-die |
| 2 | Dose cliff | λ 5–200 e⁻/px | F1 vs dose; lowest λ with F1 ≥ 0.9 | F1 ≥ 0.9 at λ ≥ 50 | die-to-die; CNN |
| 3 | Zero-shot geometry | calibrate/train on A; test on B, C | F1 drop (points) | topology drop ≤ 5 pts | Manhattan-trained CNN |
| 4 | Misalignment | shift ±8 px + small warp | % verdicts unchanged vs zero shift | ≥ 95% | pipeline without Module 3 |
| 5 | Near-miss capture | notches/filaments 1–3 nm short | recall at fixed false-alarm rate | ≥ 80% | Δβ rule alone |
| 6 | Calibration & latency | 3 scribe patches; 1,000 patches CPU | ms/calibration; ms/patch per module | calibration < 50 ms | CNN, same hardware |

Protocol rules:
- Fixed splits per family (train/val/test), fixed seeds, sample counts in config.
- Report mean and a 95% bootstrap CI; report sample counts next to every number.
- Write results to `results/expN/` as CSV + PNG + `config.json`. The README table is
  generated from these CSVs by `scripts/reproduce.py`; never type numbers into it by hand.
- If a target is missed, keep the number, keep the target, and add a one-line note on
  why in `PROGRESS.md`. Changing a target after seeing test results requires an explicit
  note saying so.
- Label every result "synthetic data".

## 7. Known bugs in the reference code (fix and test)
1. `nn.Dropout2d` on a 1-channel input zeroes the whole image (stretch denoiser only);
   use per-pixel masking.
2. Phase correlation is integer-only and its shift sign is untested.
3. Module 2 stacks patches before alignment.
4. Percentile normalisation + fixed τ = 0.35 can miss a dim bridge (both components keep
   lifetime ≥ τ) and can fake bridges from edge bloom.
5. Topology compared without a boundary margin.
6. Nuisance injection can change topology by chance (σ near half-width); labels must
   be checked against W.

## 8. Engineering rules
- Python ≥ 3.10; deps: numpy, scipy, scikit-image, gdstk, gudhi, torch (CNN only),
  matplotlib, pandas, pytest. Ask before adding anything else.
- CPU-first. A `--quick` flag must run every experiment end-to-end in a few minutes with
  small sample counts; the full run may take longer.
- Determinism: one seed in config drives everything (numpy Generator, torch, SVD
  sketch). Same config → identical CSVs.
- Tests: at least one per module and one per defect class, including invariance tests
  (LER-only sample → NUISANCE; shifted sample → same verdict) and testbed label checks.
  Run `pytest -q` before every commit; do not commit red tests.
- Type hints and short docstrings that state units (nm, px, e⁻/px).
- Small commits, one milestone at a time, with clear messages.
- Never fabricate, smooth, or hand-edit results. Never silence a failing test to pass.

## 9. Workflow
1. Explore: read all docs and the existing repo; run whatever exists.
2. Write `PLAN.md`: audit of existing code, milestone list mapped to Section 3, design
   choices still open (Section 5.2), and risks. **Stop and wait for approval.**
3. Per milestone: implement → tests → run → update `PROGRESS.md` (done, measured numbers,
   surprises, next) → commit.
4. Ask before: changing scope, adding dependencies, restructuring the repo, or changing a
   pre-registered target.
5. Final: `python scripts/reproduce.py` regenerates all results and the README table;
   the notebook walks one tile through every stage and shows each verdict with its reason.

## 10. Definition of done
- `pytest -q` passes; `python scripts/reproduce.py --quick` and the full run both finish.
- README has: one-paragraph thesis, pipeline diagram (ASCII is fine), how to run, the
  generated results table with baselines and CIs, limitations (sim-to-real gap, τ
  sensitivity, singleton layouts, scribe-to-die transfer, patch boundaries).
- No number anywhere in the repo comes from the docs' scorecards.
