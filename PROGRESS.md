# PROGRESS.md — Nanovate SEM Topology Defect Verifier

Status: **M1-M8 done (v1). v2 post-hoc fixes done (`docs/V2_FIXES.md`); `pytest -q` = 170 passed.**
Contract: `AGENTS.md` (authoritative precedence: AGENTS.md > v4.0 Part A > protocol > v3.0).
All results reported here are from **synthetic data**.

---

## Milestone M1: Scaffold + Synthetic SEM Testbed

### 1. What was built and delivered
- **Scaffold & Configuration:**
  - `pyproject.toml`: Editable project package definition.
  - `src/nanovate/config.py`: Dataclasses for `LayoutCfg`, `NuisanceCfg`, `DefectCfg`, `SEMCfg`, `TestbedCfg`, plus `save_config`. Fixed defaults for forward model and DRC rules.
  - `src/nanovate/rng.py`: Deterministic seeding using `np.random.SeedSequence(entropy, spawn_key)` and `child_rng`, guaranteeing that sample draws are decoupled across workers and splits.
- **Topology Core Utilities (`src/nanovate/topo_utils.py`):**
  - Connectivity convention: conductor is 4-connected (vertex-based cubical complex matching `scipy/skimage connectivity=1`), dielectric is 8-connected dual (holes are 8-connected dielectric regions not touching the tile border). Pinned by test.
  - Margin handling (`Window`, `relevance_region`, `betti_local`, `delta_betti`): Boundary-aware feature counting where features outside the central window cannot enter or leave through an edge displacement $< \text{tol}$ px.
- **Testbed Package (`src/nanovate/testbed/`):**
  - `layouts.py`: 3 layout families:
    - **A (Manhattan):** 14 nm half-pitch lines with designed line-end cuts and straps joining line pairs.
    - **B (Curvy):** Smooth undulating spline wires via `gdstk.FlexPath` with minimum width/spacing $\ge 14$ nm.
    - **C (Via):** Round/square pillar arrays with wide rails.
    - Fast scanline rasterizer `fill_polygon` matching `skimage.draw.polygon` at 1 nm/px.
  - `perturb.py`:
    - Stationary Ornstein-Uhlenbeck process (AR(1) for open segments; circular circulant embedding for closed contours to avoid closure seams), $\sigma = 0.5\text{--}2.5$ nm, $\xi = 20$ nm.
    - Voronoi grain contrast field on conductor (up to 30%).
    - Level-set edge normal displacement with anti-aliasing and speck removal.
    - Defect injectors with buffer constraints keeping them out of tile margin: bridge capsule, open cut, particle blob in dielectric, near-miss notch, near-miss filament (1–3 nm short of neighbour).
  - `sem.py`:
    - Physics-based forward model deliberately outside Module 1 linear feature span: saturating bloom $a_2 \cdot s \cdot \tanh(|\nabla(G_{\sigma_\text{edge}} * W)| / s)$, anisotropic scan beam $\sigma_\text{beam} = (1.6, 1.2)$ nm, 2nd-order polynomial charging ramp, Poisson shot noise (dose $\lambda \in [5, 200]\ e^-/\text{px}$) + Gaussian read noise, random sub-pixel/pixel shift.
  - `labels.py`:
    - Strict ground-truth labels derived solely from topological difference $\Delta\beta = (\Delta\beta_0, \Delta\beta_1)$ of wafer $W$ vs CAD on the central window (`consistent()`), never from injection intent.
  - `dataset.py`:
    - Deterministic `make_sample` and `make_stack` (K=32 same-layout dies with independent nuisance/defects/shifts).
  - `scribe.py`:
    - 3 isolated scribe features per layer (pad, wide line, cross) under the identical SEM forward model.
- **Reporting & Verification (`scripts/testbed_report.py`):**
  - Generates `results/testbed/resamples.csv`, `results/testbed/gallery.png`, `results/testbed/config.json`.
- **Test Suite (`tests/test_testbed.py`):**
  - 33 tests passing with 0 warnings.

---

### 2. Measured Numbers

#### Test suite:
- `pytest -q`: **33 passed** in ~18 s (CPU, M3 Pro).
- Tests cover:
  - Fill polygon equivalence to `skimage.draw.polygon` (100 random polygons).
  - Corner touch is not a short (4-conn conductor: 2 components, 0 holes).
  - Hole counting and dual connectivity (8-conn dielectric).
  - Margin rule ignores slivers and margin holes.
  - OU process variance and correlation length ($e^{-1}$ at lag $\xi$).
  - Layout DRC cleanliness (`min_space_ok`) and seed determinism across all 3 families.
  - Ground-truth topology agreement across all 6 defect classes $\times$ 3 families $\times$ 3 dies ($n=54$ samples).
  - Filament gap verification ($1.0 \le \text{gap} \le 3.75$ nm in $W$).
  - Defects strictly contained within central window (outside margin).
  - SEM image units (mean counts track $\lambda \cdot \bar{I}$, variance tracks Poisson + read noise).
  - Stack sharing identical CAD with independent nuisance.
  - Scribe patch generation and bounds.
  - Resample tracking at high LER $\sigma = 2.5$ nm.

#### Resample statistics (`scripts/testbed_report.py --quick`, seed=0):
- Sample count: $N = 720$ total evaluations across 3 families $\times$ 5 LER $\sigma$ levels $\times$ 6 kinds $\times$ 8 layouts.
- **Total failed samples:** **0 / 720 (0.0% failure rate)**.
- Summary of resamples per sample and reject breakdown:

| Family | LER $\sigma$ (nm) | Kind | $n$ | Failed | Resamples/sample | LER topology rejects | Defect topology rejects | Latency (ms/sample) |
|---|---|---|---|---|---|---|---|---|
| Manhattan | 0.5 | nuisance | 8 | 0 | 0.00 | 0 | 0 | 89.1 |
| Manhattan | 0.5 | bridge | 8 | 0 | 0.00 | 0 | 0 | 89.9 |
| Manhattan | 0.5 | open | 8 | 0 | 0.00 | 0 | 0 | 92.3 |
| Manhattan | 0.5 | particle | 8 | 0 | 0.00 | 0 | 0 | 91.8 |
| Manhattan | 0.5 | notch | 8 | 0 | 0.00 | 0 | 0 | 92.4 |
| Manhattan | 0.5 | filament | 8 | 0 | 0.00 | 0 | 0 | 92.8 |
| Manhattan | 1.0 | nuisance | 8 | 0 | 0.00 | 0 | 0 | 94.0 |
| Manhattan | 1.0 | open | 8 | 0 | 0.125 | 0 | 1 | 106.2 |
| Manhattan | 1.0 | notch | 8 | 0 | 0.250 | 0 | 2 | 111.8 |
| Manhattan | 1.5 | open | 8 | 0 | 0.250 | 0 | 2 | 113.7 |
| Manhattan | 1.5 | notch | 8 | 0 | 0.125 | 0 | 1 | 109.2 |
| Manhattan | 2.0 | nuisance | 8 | 0 | 0.375 | 3 | 0 | 116.8 |
| Manhattan | 2.0 | notch | 8 | 0 | 0.500 | 3 | 1 | 132.7 |
| Manhattan | 2.5 | nuisance | 8 | 0 | 1.000 | 8 | 0 | 155.7 |
| Manhattan | 2.5 | notch | 8 | 0 | 0.875 | 4 | 3 | 159.6 |
| Manhattan | 2.5 | filament | 8 | 0 | 0.500 | 4 | 0 | 131.8 |
| Curvy | 0.5–1.0 | all | 8 | 0 | 0.00 | 0 | 0 | ~180 |
| Curvy | 2.0 | open/fil | 8 | 0 | 0.125 | 1 | 1 | ~210 |
| Curvy | 2.5 | open | 8 | 0 | 0.250 | 1 | 1 | 224.4 |
| Curvy | 2.5 | notch | 8 | 0 | 0.375 | 0 | 3 | 275.4 |
| Via | 0.5–1.5 | all | 8 | 0 | 0.00 | 0 | 0 | ~80–90 |
| Via | 2.0 | particle | 8 | 0 | 0.375 | 3 | 0 | 122.9 |
| Via | 2.5 | nuisance | 8 | 0 | 0.125 | 1 | 0 | 115.5 |
| Via | 2.5 | particle | 8 | 0 | 0.750 | 6 | 0 | 156.5 |
| Via | 2.5 | notch | 8 | 0 | 0.500 | 1 | 3 | 145.7 |
| Via | 2.5 | filament | 8 | 0 | 0.375 | 3 | 0 | 127.4 |

---

### 3. Surprises and Bug Fixes

1. **Extension to Bug #5 (Boundary margin for enclosed holes):**
   - *Discovery:* Initially on Manhattan val layout 2 at $\sigma = 0.5$ nm, 51 consecutive resample rejections occurred for nuisance samples due to $\Delta\beta_1 = 1$. Investigation revealed that two straps had formed an enclosed dielectric hole located at rows 240–253 (entirely outside the central window $[16, 240)$). When 0.5 nm LER was applied, the conductor line at row 239 roughened by 1 px, causing the margin hole to touch row 239.
   - *Root cause:* `relevance_region` previously only dilated and subtracted outside conductor components (`cad_out`), leaving outside holes unprotected; and `betti_local` hardcoded `inner & ~m` for holes instead of evaluating within the relevance region `reg & ~m`.
   - *Fix:* Extended `relevance_region` to identify outside CAD holes (`holes_out`) and subtract `dilate(holes_out, tol)`, and updated `betti_local` to evaluate `reg & ~m`. This brought $\sigma = 0.5$ nm resamples to exactly 0 across all layouts and fixed the regression cleanly.
2. **Physical Confirmation of Bug #6 (Survivor bias and LER-induced topology changes):**
   - At high roughness ($\sigma = 2.5$ nm) on 14 nm lines, random OU fluctuations frequently pinch traces or bridge adjacent lines. On Manhattan nuisance samples at $\sigma = 2.5$ nm, exactly 8 out of 8 initial attempts suffered topology-changing LER and were caught by the resample validator (`ler_topology_rejects = 8`). All samples were successfully resampled with honest accounting recorded in `resamples.csv`.

---

### 4. What's Next
- **Milestone M2:** Module 4 — Topology verdict + bottleneck-distance near-miss rule (`modules/topology.py`, `modules/fields.py`).
  - GUDHI cubical persistence (vertex construction for 4-connectivity).
  - Conductor probability fields (options a and b with oracle levels for calibration baseline).
  - Lifetime threshold filtering $\tau$.
  - Bottleneck distance against reference diagram.
  - Reason payload dictionary and unit tests.


---

## Milestone M2 (in progress): Module 4 topology verdict

### Done
- `modules/fields.py` (options a: percentile, b: calibrated levels; b uses **oracle** levels until M3), `modules/topology.py` (GUDHI cubical PH, Betti at lifetime >= tau, bottleneck H0/H1, verdict ladder + reason dict). NEAR_MISS is disabled until SDF energy exists (M4).
- `pytest -q`: 45 passed.

### Surprises / bugs found while getting green
1. **Verifier ignored the label's margin rule (Bug #5 in the verifier).** A plain crop counted wire slivers (5 px and 27 px) pushed across the window edge by LER, giving a false KILLER_OPEN/PARTICLE on a nuisance sample (d_beta0 = +2) although the ground-truth label is nuisance. Fix: compute on the whole tile with the field replaced by CAD outside `relevance_region(cad, win)`, the same region the labels use.
2. **Bug-4 test premise was wrong.** A bridge of conductor level c merges wires at filtration 1 - c, so it is detected iff c > 1 - tau. At tau = 0.45 a 0.5-contrast bridge is *missed*; the test now uses c = 0.6 (tau 0.35 misses, 0.45 detects). Consequence: tau directly trades dim-bridge recall against bloom fake-bridges.

### Not yet done (M2 exit criteria)
- False-killer rate vs dose for field options (a)/(b) on Manhattan val; choose field and tau/d*/e* on Manhattan val only, document why.
- Tests so far run on noise-free expected images and a few LER samples only; no noisy-image numbers measured yet.

### M2 early-risk probe (`scripts/probe_m2.py`, Manhattan val only, seed 0, oracle levels, synthetic data)
Sweep: dose {5,10,20,50,100,200} e-/px x field {percentile, calibrated} x pre_smooth {0, 1 px} x tau 0.30-0.70.
Cells: n = 16 nuisance, n = 48 killer (8 layouts x 2 dies x {bridge, open, particle}) per (field, smooth, tau, dose). Small n: indicative only, no CIs yet.
Outputs: `results/m2_probe/{probe.csv, summary.csv, config.json}`. No tau chosen yet.

Findings (measured):
1. **Unsmoothed fields are unusable at dose <= 50**: calibrated/no-smooth false-killer rate is 1.0 across the tau grid (killer-correct 0.33 = everything flagged); shot-noise pixels form long-lived components. Percentile/no-smooth is similar until dose >= 100. Some denoising before Module 4 (Module 2, or at least pre-smoothing) is required, not optional.
2. **No tau gives low false-killer and high recall together.** Calibrated, pre_smooth = 1, dose >= 50: false-killer falls from ~0.44 (tau 0.30-0.40) to ~0.06-0.12 (tau 0.65-0.70), while killer-correct falls from ~0.73 to ~0.12-0.44. At dose 5-10 the false-killer rate is 0.12-1.0 for every tau.
3. **Failures are systematic, not noise.** At dose 200, calibrated, smooth 1, tau 0.5: bridge recall 16/16, but open correct 8/16 (6 called NUISANCE, 2 called BRIDGE) and particle correct 7/16 (7 NUISANCE, 2 BRIDGE); nuisance false-killer 5/16, all in layouts 1, 2, 6, 7 (layouts 0, 3, 4, 5 are clean at dose >= 50).
4. Percentile + pre_smooth 1 reaches false-killer 0.00 at dose 50 / tau 0.55 but killer-correct there is only 0.65, and it degrades again at dose >= 100. Not a robust choice.

Not yet explained (to investigate before choosing a field/tau): why opens/particles are missed and why layouts 1, 2, 6, 7 raise false killers (candidates: bloom/blur filling short cuts and narrow gaps, shift vs. relevance-region mismatch, charging ramp vs. oracle levels). The Bug #4 trade-off is confirmed on the full pipeline: tau cannot be a single fixed constant across dose.

### M2 diagnosis of probe failures (evidence from scratch scripts; Manhattan val, dose 200, tau 0.5, oracle levels, n = 16 nuisance / 16 open / 16 particle)
Evaluated on the noise-free `expected` image first, so these are not shot-noise effects. Four separate mechanisms:
1. **Optical resolution limit (not a verifier bug).** Particles whose gap to a wire is ~3-4 px merge with the wire at P > 0.5 (e.g. layout 0 die 0: isolated 57-px particle in W, 4 px from a wire, fused with it in the image). Opens with cut length <= ~4.5 px are not resolved (P at the cut stays above 1 - tau). Roughly 6/16 opens and 7/16 particles are missed on this basis; some ground-truth "killers" are simply sub-resolution at this blur/bloom.
2. **`pre_smooth = 1` makes (1) worse.** Opens of length 3.7-4.9 px are detected noise-free with no smoothing and lost with sigma = 1 (layouts 0, 3, 4, 7). Smoothing is needed at low dose, so denoising must come from Module 2 (low-rank stack), not a blur on top of the beam blur.
3. **Region seam is narrower than the optical reach.** `relevance_region(tol=4)` keeps only 4 px clear of out-of-window wires, but blur + bloom reach ~5 px and the acquisition shift is up to 3 px, so a neighbouring wire leaks into the region (layout 7 die 0, shift 0: 5-px sliver at row 243 with no partner in W). Nuisance false killers with `region_tol` (noise-free, 16 samples): tol 4: 5 random-shift / 3 zero-shift; tol 6: 3 / 2; tol 8: 1 / 1; tol 10: 2 / 1. Not eliminated.
4. **Splicing image into CAD at the region boundary creates artifacts.** At tol 8-10 the remaining false killers are new holes (d_beta1 = +1..+4), consistent with a seam between image content and CAD fill. Delta-beta is also a *net* count, so a missed particle (+1) and a spurious merge (-1) can cancel to (0, 0), hiding errors.
Shift alone flips verdicts (layouts 1, 2, 6 false killers appear only with the random shift; 5, 7 flip the other way), i.e. Module 3 alignment matters for the verdict, as Experiment 4 anticipates.
Layout 5 die 1 (shift 0, d_beta1 = +1) has no component mismatch at P > 0.5 and is unexplained.

Open code change: `verify_topology(..., region_tol=4)` parameter added (default unchanged; no tol chosen).
Candidate next steps (not done): (a) replace splicing by counting only persistence pairs whose birth/death cell lies inside the relevance region (GUDHI cofaces), no seam; (b) choose `region_tol` and tau jointly on Manhattan val after Module 3/2 exist; (c) report sub-resolution killers separately (resolution limit) instead of counting them against the verifier.

### Attempt: region-restricted persistence pairs via GUDHI cofaces -- blocked (no code changed)
`CubicalComplex.cofaces_of_persistence_pairs()` raises for the `vertices=` constructor ("only makes sense for a complex initialized from the values of the top-dimensional cells"). The top-dimensional-cell constructor makes sublevel sets unions of closed squares, i.e. 8-connected conductor, which reverses the project's pinned convention (conductor 4-connected, dielectric 8-connected; corner touch must not short). Using it would need a duality re-derivation (conductor components via H1/H0 of the dielectric side, with tile-border effects) and re-validation of every M2 test.
Further, localisation is ill-posed on the CAD reference: CAD is a binary plateau, so the death vertex of a wire that merges at the dielectric level is an arbitrary tie-break, while a wire's birth vertex (its max-P pixel) lies anywhere along the wire; only H1 (holes) and finite H0 merges in the image have a stable location.

### Region-local pairs (`modules/localpers.py`) vs splice -- probe comparison (`scripts/compare_region_modes.py`)
Implemented own union-find H0 + dual union-find H1 (GUDHI cofaces unavailable for the vertex construction); `verify_topology(region_mode="pairs"|"splice")`. Validated: finite H0/H1 intervals equal GUDHI's exactly (incl. tied values); binary counts equal `betti_local` on testbed samples for all 6 kinds. 66 tests pass.
Comparison: Manhattan val, seed 0, oracle levels, synthetic; n = 16 nuisance, 48 killer per cell (false-killer granularity 1/16 = 0.06). Outputs `results/m2_region_modes/{rows,summary}.csv`. Dose 5/10 not rerun (earlier probe: unusable).
Results (calibrated field unless noted):
1. **`region_tol` matters more than the counting mode.** With pre_smooth 1, tau 0.45, false-killer at dose 20-200: splice tol 4 = 0.31-0.38 -> tol 8 = 0.06 (flat); pairs tol 4 = 0.25-0.31 -> tol 8 = 0.06-0.19.
2. **Pairs is not a clear win.** With smoothing, splice tol 8 has the lowest, dose-flat false-killer (0.06), pairs tol 8 is slightly higher (0.06-0.19) but with higher killer-correct (0.60-0.73 vs 0.58-0.65). Without smoothing at dose 200, pairs wins (tau 0.55, tol 8: false-killer 0.00 / killer-correct 0.71 vs splice 0.12 / 0.62), but unsmoothed fields flag everything at dose <= 100 in both modes.
3. Percentile + smooth 1 is not robust: at tau 0.55 pairs false-killer is 0.94 at dose 20 while splice is 0.50-0.69.
4. Killer-correct stays <= ~0.79 everywhere, consistent with the sub-resolution killers found in the diagnosis (not a counting problem). Pairs tol 8 false killers at dose 200 (3/16) are not yet explained.
Not decided: field, tau, region_tol, region_mode (defaults unchanged: splice, tol 4). Next evidence needed: Module 1 levels (replace oracle), Module 2 denoising (replace pre_smooth), and classification of sub-resolution killers.

---

## Milestone M3: Module 1 scribe calibration (`modules/calibration.py`, `tests/test_calibration.py`, `scripts/calibration_report.py`)
Status: implemented; `pytest -q` = 75 passed (9 new). Results in `results/m3_calibration/`. Synthetic data; Manhattan/curvy/via **val** splits; seeds 0-2.

### Built
5-feature closed-form ridge (features standardised, intercept unpenalised) on the 3 scribe patches; levels mu_diel = w0, mu_cond = w0+w1+w2; noise model Var = alpha*mu + sigma2; generalised Anscombe; optional per-patch ramp removal (off by default); sub-pixel scribe registration (`estimate_shift`, sign tested against a known shift).

### Measured (n = 3 scribe seeds per dose; transfer n = 18 val patches per family/dose cell)
| Quantity | Result |
|---|---|
| mu_diel / mu_cond vs simulator's interior levels | within 0.4% at doses 20-200 (large flat interiors pin them) |
| Scribe-to-die transfer error \|\|T(C) - E[I]\|\|/\|\|E[I]\|\| (registered) | curvy 0.107, manhattan 0.118, via 0.124; independent of dose (relative) |
| same, no scribe registration | 0.121 / 0.134 / 0.139 (registration helps ~1.5 points) |
| same, with per-patch ramp removal | 0.105 / 0.116 / 0.122 (gain 0.001-0.003: not worth enabling) |
| Noise gain alpha (true 1) | 0.95-1.00 across doses |
| Read variance sigma2 (true 1) | 0.7-1.4 (noisy across seeds; 0.0-2.05 per seed) |
| Calibration time, 3 patches, CPU | 80 ms with registration; 27 ms without. **Target < 50 ms (Exp 6) is MISSED with registration**, met without |
| M2 verdicts, calibrated vs oracle levels (calibrated field, smooth 1, splice, tol 8, tau 0.45; not tuned) | false-killer 0.06 for both at every dose; killer-correct 0.60-0.67 vs 0.58-0.65 (n = 16 / 48 per cell). Calibrated levels replace the oracle with no measurable loss |

Transfer error is 11-12%, i.e. not ~0: the deliberate model mismatch (anisotropic beam, saturating bloom, grain, ramp) keeps the 5-feature span from being circular (risk R1 not triggered).

### Rulings / surprises
1. **Noise model: neighbour differences, not binned residuals** (deviates from PLAN M3 wording). A plain residual fit absorbed charging ramp, grain and template error: alpha rose from 1.07 to 1.88 with dose (true 1). Horizontal neighbour differences in flat pixels (|grad G1*C| < 1e-3) cancel smooth structure; alpha now 0.95-1.00. Test builds a ramp + blocky-grain case where the plain fit is > 1.5.
2. **Binning bug found by profiling.** Predicted means form exact plateaus, quantile edges collapsed, and a "few distinct values" fallback created ~14k bins: 0.78 of 0.86 s of calibration time. Fix: fallback only for <= 4*nbins distinct means. (An earlier guess that cubic `ndi.shift` was the cost was wrong; I still replaced it by a bilinear shift of the binary CAD before blurring, which is cheaper.)
3. **Registration helper lives in calibration.py** (plan had alignment only in Module 3). Scribe patches carry random sub-pixel shifts; without registering them the fit is biased (transfer error +1.5 points). Plain mean-removed cross-correlation (`normalization=None`): phase whitening failed on smooth/noisy inputs, a Hann window hurt on edge-touching features. Module 3 owns in-die alignment.
4. Test change, not a code change: the synthetic ramp in the neighbour-difference test was made zero-mean within each half, so the template mean stays right (a ramp with nonzero per-region mean shifts the true mean away from the template and inflates the slope; that is a mean-assignment effect, not noise).

### Caveats
- Transfer error is measured at shift 0 against the noise-free expected image (includes the charging ramp). With real in-die shifts this needs Module 3.
- Levels are calibrated at the same dose as the die patches; the dose sweep (Exp 2) must recalibrate per dose.
- sigma2 is poorly determined at high dose (read noise is small next to shot noise).

### Next
M4: Module 3 (sub-pixel alignment with sign test, CAD SDF tension energy), which also feeds the near-miss rule in Module 4 and the shift-sensitivity seen in the M2 probe.

---

## Milestone M4: Module 3 alignment + SDF tension energy (`modules/align_sdf.py`, `tests/test_align_sdf.py`, `scripts/align_sdf_report.py`)
Status: implemented; `pytest -q` = 96 passed (21 new). Results in `results/m4_align_sdf/`. Synthetic data, Manhattan val, seed 0, Module-1 calibrated levels, fixed rule (smooth 1, splice, region_tol 8, tau 0.45; nothing tuned). Near-miss branch of `verify_topology` still disabled: d*, e* are not chosen (M7).

### Built
`align_to_template` (cross-correlation against T(C), 20x sub-pixel, 3 passes, cubic `ndi.shift`; sign convention: `ndi.shift(image, d) ~ template`, d ~ -shift, tested both signs and round trip), `cad_sdf` (EDT, px = nm, dielectric > 0), `tension_energy` = |S| exp(gamma min(|phi|, cap)), `max_tension_energy` (S = smoothed residual in contrast units; gamma = 0.2, cap = 10 px fixed a priori). No Langevin step.

### Measured
1. **Estimator accuracy** (template = the wafer's own noise-free image, shifts ~±8 px, 4 shifts x 3 families): max error 0.05 px at dose 200 (manhattan/curvy/via), <= 0.20 px at dose 5 -> inside the plan's <0.1 px (high dose) and <0.25 px (lambda = 5) tolerances.
2. **Alignment against the CAD-derived template misses the <0.1 px tolerance** (n = 4 shifts per cell, 3 passes): manhattan max 0.40-0.45 px, curvy 0.65 px (dose 200) / 0.90 px (dose 5), via 0.20 px. The floor is the CAD/wafer mismatch (LER), unchanged by extra passes; at true shift 0 the estimate is already off by up to ~0.45 px. In the full report the median error against T(C) is 0.33-0.45 px. Target kept, miss recorded.
3. **Bug found while testing:** single-pass circular correlation biased the peak in proportion to the shift (content entering from outside the frame): error 0.1-0.5 px even with a perfect template. Iterating on the progressively aligned image fixed it (Manhattan 0.15 -> 0.05 px, curvy 0.50 -> 0.05 px at dose 200). Zero-padding made it worse; my masked-correlation attempt returned nonsense (not pursued); Hann windowing and pre-smoothing hurt on testbed images.
4. **Near-miss separation (risk R3 dry run), n = 16 per kind/dose.** Column `dbeta0` in the report = fraction with Delta-beta = (0,0) (misnamed). Notch: 94-100% keep Delta-beta = 0. **Filament: only 25-31% keep Delta-beta = 0; the rest read as bridges** (gap 1-3 px is below the optical resolution), so most filaments are unreachable by a near-miss rule. Among Delta-beta = 0 samples, near-miss (notch|filament, n = 19-21) vs nuisance (n = 15-16): AUC bottleneck 0.74 (dose 50) / 0.79 (200); AUC max SDF energy 0.98 / 1.00. Caveat: energy separates "a defect is present" from "none" (nuisance has no defect), not near-miss from a harmless nuisance of similar size; it is not a topology criterion.
5. **Experiment-4 preview** (dose 50, shifts ±8 px incl. diagonals, no warp; verdict vs the same wafer's zero-shift acquisition; n = 128 per mode): **with Module 3: 85.5% unchanged; without: 52.7%.** Target >= 95% is **missed**. By kind with alignment: bridge 100%, nuisance 95%, open 78%, particle 69% (the sub-resolution killers flip with sub-pixel phase). This preview excludes the small warp that Exp 4 adds.

### Notes
- Alignment is to T(C) as planned; the sample-level floor means topology sees ~0.4 px residual misregistration, which `region_tol` (8) must absorb.
- Energy parameters (gamma, cap, smooth) are a-priori constants, not selected on val.

### Next
M5 (Module 2): aligned K = 32 stack + rank-2 randomized SVD + MAD, with singleton fallback. It is also the denoiser that should replace `pre_smooth` (which hid sub-resolution opens).

---

## Milestone M5: Module 2 low-rank stack (`modules/lowrank.py`, `tests/test_lowrank.py`, `scripts/lowrank_report.py`)
Status: implemented; `pytest -q` = 106 passed (10 new). Results in `results/m5_lowrank/`. Synthetic data, Manhattan val, seed 0, Module-1 calibrated levels, fixed M2 rule (splice, region_tol 8, tau 0.45; nothing re-tuned for the stack field).

### Built
`denoise_stack`: align each die to T(C) (Module 3, bug #3), K = 32 matrix, seeded rank-2 randomized SVD (oversample 2, 2 power iterations), per-pixel MAD sigma across dies, threshold, field = L + S; `denoise_singleton` (median 3x3 + Gaussian 1 px) flagged `singleton=True`. Tested: randomized SVD vs full SVD (< 1e-3), seed determinism, defect in its own die only, singleton flag, soft vs hard vs support thresholds, residual smoothing as a matched filter.

### Measured
1. **Alignment matters at the Exp-4 range.** Low-rank residual ||D - L||/||D|| on noise-free dies with random ±8 px shifts: unaligned 0.211, template-aligned 0.110, oracle (true shifts undone) 0.111; consensus refinement adds nothing. With the testbed's default ±3 px shifts the gain is only ~10% (0.124 -> 0.110), because rank 2 absorbs first-order shifts. The remaining 0.11 is independent LER/grain per die. (The first test, on noisy dies at ±3 px, could not see this: shot noise alone is ~0.19 of the norm.)
2. **Stack field vs per-patch smoothing in Module 4** (n = 48 clean + 36 killer dies per cell, 6 layouts x 8 clean + 6 defective dies; K = 32 with 6 defective dies per stack):

| field | dose | false-killer | killer-correct | bridge | open | particle |
|---|---|---|---|---|---|---|
| smooth1 (aligned image, Gaussian 1 px) | 20 / 100 | 0.04 / 0.02 | 0.67 / 0.64 | 1.00 / 1.00 | 0.75 / 0.58 | 0.25 / 0.33 |
| medgauss (singleton fallback) | 20 / 100 | 0.06 / 0.08 | 0.64 / 0.64 | 1.00 / 1.00 | 0.50 / 0.58 | 0.42 / 0.33 |
| stack soft k2, no smoothing (spec) | 20 / 100 | **1.00** / 0.15 | 0.33 / 0.58 | 1.00 / 0.75 | 0.00 / 0.08 | 0.00 / 0.92 |
| **stack soft k2 + residual smooth 1 (chosen)** | 20 / 100 | 0.15 / 0.00 | 0.61 / 0.64 | 0.92 / 0.92 | 0.08 / 0.08 | 0.83 / 0.92 |

   Other variants (all in `results/m5_lowrank/rows.csv`): hard k3, hard k3 + smooth, hard k4 + smooth, soft k3 + smooth, support k2/k3 + smooth: false-killer 0.00-1.00; none beats the chosen one on both false-killer and killer-correct.
3. **Net result: Module 2 does not beat per-patch smoothing for Module 4 in this testbed.** Overall killer-correct is equal within noise (0.61/0.64 vs 0.67/0.64), false-killer is slightly higher at dose 20 (0.15 vs 0.04). The error profile differs: the stack finds particles far better (0.83-0.92 vs 0.25-0.33) but loses opens (0.08 vs 0.58-0.75). Small n (12 per defect class per cell): differences of ~0.1-0.15 are within sampling noise; the particle and open gaps are not.
4. Timing: ~16 ms/patch for a K = 32 stack including alignment (CPU).

### Rulings / surprises
1. **Default deviates from the spec** (spec: per-pixel MAD soft threshold on the residual): `denoise_stack(resid_smooth=1.0)` smooths each die's residual by 1 px before the threshold. At 0 (spec) isolated noise pixels survive and Module 4 sees many spurious components/holes (dose 20: false-killer 1.00, up to 6-8 extra holes per patch). Chosen on Manhattan val from 8 variants listed above, so it is a val-selected setting; the library helper `sparse_part` keeps resid_smooth = 0 as the spec baseline.
2. **Opens are lost because narrow defects are attenuated twice** (smoothing, then soft shrinkage): at the open the aligned image has P down to 0.13-0.26 but L + S restores it to a mean 0.63-0.92 (S supplies -3..-16 counts of the ~-35 needed). My follow-up hypothesis (keep the raw residual amplitude on the smoothed-residual support, `support` mode) was **rejected**: false-killer 1.00 everywhere, because per-die LER makes the smoothed residual exceed the threshold along every edge, and the raw amplitude there puts pixel noise back. Not pursued further.
3. A bug of mine while adding the options (a replace matched the wrong occurrence and truncated the module) was caught immediately by the import error in the tests and repaired by rewriting the file.
4. The Anscombe transform is not used inside Module 2: the per-pixel MAD floor already tracks the heteroscedastic noise. It remains available in `calibration.py`.

### Caveats
- Stack stats here assume 6 of 32 dies defective (19%); a defect rate near 50% would break the MAD floor. Dies are all same-layout, aligned to T(C) which carries the ~0.4 px CAD/wafer floor.
- tau/region_tol/levels were not re-tuned for the stack field; opens may need a different rule.

### Next
M6 baselines: die-to-die (single and double difference) and the Manhattan-only CNN, with thresholds chosen on Manhattan val only.

---

## Milestone M6: baselines (`baselines/die_to_die.py`, `baselines/cnn.py`, `scripts/baselines_report.py`)
Status: code + tests done (11 new tests pass: D2D flags an obvious bridge, not a clean die or re-acquisition; double-difference cancels a defect in one reference; val-only threshold fit refuses other splits; CNN forward shape, seeded determinism, tiny-batch overfit, val-only threshold). The first-run numbers below were degenerate; final val numbers are in the Results section.
D2D aligns target and references with the Module-3 aligner and applies the Module-1 GAT (z-units), so alignment and noise scaling are not confounds.

First full run on Manhattan val (synthetic; 50% killer prevalence, so "flag everything" scores F1 = 0.67), both baselines degenerate:
- **D2D with one pooled rule over doses 20/50/200**: F1 0.68, nuisance false alarm 0.71 overall; at dose 50 and 200 it flags every clean die. LER differences between dies grow relative to shot noise as dose rises, so a z-threshold cannot transfer across doses. Fix in the script (not yet rerun): fit the rule per dose (recipe tuned per condition, the best case for D2D), keep the pooled rule as a comparison.
- **CNN**: training loss 0.733 -> 0.701 (chance), i.e. untrained: 10 epochs x 18 steps on 576 mixed-dose samples is too few steps to judge. A longer schedule is to be trained off this machine (Colab); until then no CNN number is reported.

### M6: CNN training moved off this machine (Colab)
`scripts/train_cnn.py` + `notebooks/train_cnn_colab.ipynb` train the CNN on a GPU: 48 Manhattan-train layouts x 4 dies x 6 kinds = 1152 patches, random dose per sample (5-200 e-/px), 60 epochs (~2,200 steps), seed 0; the budget is fixed before any val/test CNN number exists. Weights come back as `cnn.pt` and are evaluated here with `scripts/baselines_report.py --cnn-weights` (val-chosen threshold, F1, latency). GPU training is not bit-reproducible; config, seed and loss history are saved with the weights. Not yet run.

### M6 results: baselines evaluated on Manhattan val (`results/m6_baselines/`, CNN weights `m6_cnn/cnn.pt`)
Synthetic data. Val items: 8 layouts x 2 dies x 6 kinds at doses 20 / 50 / 200 (n = 96 per dose, 48 killers = 50% prevalence, so "flag everything" scores F1 = 0.67). **These are IN-SAMPLE for the selected thresholds** (D2D: 3 parameters per dose; CNN: 1 probability threshold, 0.39); the CNN weights were trained only on Manhattan train. No test split has been used; held-out numbers come from the M7 experiments.
CNN: Colab GPU, 1152 train patches, 60 epochs, loss 0.728 -> 0.210, in-sample train accuracy 0.93 (torch 2.11 on Colab vs 2.14 here; GPU training is not bit-reproducible, config and loss history are saved).

| method (Manhattan val) | dose 20 | dose 50 | dose 200 | all (n = 288) | nuisance false alarm (all) | recall bridge / open / particle (all) |
|---|---|---|---|---|---|---|
| D2D single, per-dose rule | 0.72 | 0.68 | 0.69 | 0.70 | 0.08 | 0.98 / 0.12 / 0.71 |
| D2D double, per-dose rule | 0.70 | 0.70 | 0.69 | 0.69 | 0.33 | 1.00 / 0.46 / 0.90 |
| D2D single, one pooled rule | 0.72 | 0.67 | 0.67 | 0.68 | 0.71 | 1.00 / 0.73 / 0.94 |
| D2D double, one pooled rule | 0.70 | 0.67 | 0.67 | 0.68 | 0.67 | 1.00 / 0.71 / 0.98 |
| CNN ([image, CAD], Manhattan train only) | 0.90 | 0.88 | 0.91 | 0.90 | 0.00 | 0.90 / 0.83 / 0.90 |

(values in the dose columns are F1.) Latency, this CPU: CNN 2.3 ms/patch at batch 1, 4.3 ms/patch at batch 64; D2D single including alignment + GAT + stats 35 ms/patch (not yet timed under the Exp-6 protocol).

Observations:
1. **Die-to-die is weak on this testbed even when tuned per dose** (F1 0.68-0.72, barely above the flag-everything 0.67): LER and grain differences between dies are as large as the defect signal. Opens are almost invisible to it (recall 0.06-0.25 per dose for the single-difference rule); a rule that finds them (double, dose 200) does so by flagging 81% of clean dies. A single pooled rule is useless above dose 20 (false alarm 1.00), confirming that the threshold does not transfer across doses.
2. **The CNN is a strong in-distribution baseline** (F1 0.88-0.91, zero nuisance false alarms on val) and, unlike our verifier's weak spots, it finds opens (0.83). It sees the CAD channel by design (PLAN Q5). Whether it holds on curvy/via layouts is Experiment 3.
3. The comparison with the topology verifier is not made here: M7 evaluates all methods on the same held-out test samples with bootstrap CIs.

### Next
M7: experiments 1-6 on the held-out test split with CIs (`experiments/exp1..exp6.py`, `scripts/reproduce.py`, README table).

---

## Milestone M7: pipeline, config selection, Experiments 1-6 (`pipeline.py`, `experiments/`, `scripts/reproduce.py`)
Status: implemented and **run in full** (`python scripts/reproduce.py`; the README results block is generated from `results/exp*/*.csv`). `pytest -q` = 157 passed (incl. smoke tests of every experiment on a tiny config and a two-run identical-table check for Exp 1/3/4/5). Synthetic data; Manhattan val for every selection; held-out **test** split for every reported number; 95% percentile-bootstrap CIs (1,000 resamples); n beside every number (see README for the full tables).

### Built
`pipeline.py` (singleton path: align -> template-anchored field -> topology -> SDF energy, with reason, `singleton` flag and per-module ms; stack path via Module 2), field option (c) template-anchored residual, smooth warp for Exp 4 (`warp_amp`, 0 = bit-identical to before), `experiments/{common,harness,select_config,report,exp1..exp6}.py`, `scripts/reproduce.py [--quick]`, README generated tables (idempotent, tested).

### Verifier configuration (selected on Manhattan val, frozen; `results/calib/decisions.json`)
Pre-registered rule: max killer F1 pooled over doses 20/50/200 subject to nuisance suppression >= the best percentile-field candidate (n = 288 val patches, 144 killers, 48 nuisance). Chosen: **template-anchored field, splice, tau 0.35, pre-smooth 1 px, region_tol 8** (val F1 0.847, suppression 0.917; floor from option (a): 0.708). Best per option: template 0.847/0.917; calibrated 0.775/0.729; percentile 0.788/0.708. d* = 0.088, e* = 2.39 so that NEAR_MISS fires on <= 5% of val nuisance (val: 2.1%; val near-miss recall 64.6%, n = 96). Option (c) injects a CAD prior; its recall is inside the F1 above.

### Results vs pre-registered targets (verifier; targets are hypotheses, kept as stated)
| Exp | Target | Measured (test, synthetic) | Met |
|---|---|---|---|
| 1 | >= 90% suppression and >= 90% bridge recall at LER sigma <= 1.5 nm | worst cell (sigma 1.5, grain 30%): suppression 46% [25, 67], bridge recall 79% [62, 96] (n = 24 per cell) | **NO** |
| 2 | F1 >= 0.9 at lambda >= 50 e-/px | F1 0.81-0.85 at lambda >= 50 (n = 144 per dose); never reaches 0.9 | **NO** |
| 3 | F1 drop A -> B/C <= 5 pts | curvy -3.7 [-12.7, +5.7], via -7.3 [-15.8, +1.6] pts | yes (point estimate; CI upper bound for curvy is +5.7) |
| 4 | >= 95% verdicts unchanged (shift ±8 px + warp) | 94% [91, 97] with Module 3 (n = 192); 39% [32, 46] without | **NO** (CI contains 95%) |
| 5 | near-miss recall >= 80% at the fixed false-alarm rate | 92% [87, 96] (notch 83%, filament 100%; n = 72 each) at nuisance alarm 4% [0, 10] (n = 72); delta-beta alone 29% | yes |
| 6 | calibration < 50 ms | 79 ms median (100 runs) with registration; 27 ms without | **NO** |

### Comparison with baselines (same test patches, labels)
- **Exp 2 (killer F1, 50% prevalence):** verifier 0.81-0.85 at lambda >= 20; die-to-die (per-dose rules fitted on val) 0.57-0.69; CNN 0.93-0.95 at lambda >= 20. At lambda 5-10 the verifier flags every clean patch (suppression 0%) and F1 is 0.67-0.68 (the flag-everything value). The CNN reaches F1 >= 0.9 from lambda 20; verifier and die-to-die never do.
- **Exp 3:** in-distribution (Manhattan) the CNN is best (0.94 vs verifier 0.81), but zero-shot the CNN drops +9.1 [+1.9, +16.7] (curvy) and +19.9 [+10.8, +29.5] pts (via) while the verifier's F1 is 0.84 / 0.88: on via the verifier beats the CNN (0.88 [0.82, 0.93] vs 0.74 [0.66, 0.81]); on curvy they tie (0.84 vs 0.85). The verifier's "drop" is negative partly because its Manhattan F1 is its lowest.
- **Exp 1:** all methods degrade with LER. At sigma 1.5 / grain 30%: suppression verifier 46%, die-to-die 29% / 21%, CNN 92%; bridge recall 79% / 92% / 100% / 92%. At sigma >= 2.0 everyone except the CNN collapses (verifier suppression 0-12%). The verifier meets 90% suppression only at sigma <= 1.0 and grain <= 20%.
- **Latency (this CPU):** verifier ~102 ms/patch singleton path (alignment 16, SDF energy 6, topology 79), stack path ~108 ms/patch; die-to-die ~31 ms; CNN 2.7 ms (batch 1, default threads). The CNN is ~40x faster.

### Surprises and findings
1. **Exp 1's suppression miss is mostly near-miss alarms, not killers.** At sigma 1.5, 31 of the verifier's 36 nuisance failures are NEAR_MISS verdicts and 5 are killers; the pre-registered metric counts NEAR_MISS as a false alarm. The secondary metric (not flagged KILLER) is 92% [79, 100] at sigma 1.5 / grain 30%, equal to the CNN, and 75-83% even at sigma 2.0. d*/e* were frozen on val at the nominal LER (sigma 1.0), and bottleneck/energy grow with roughness, so the near-miss rule does not transfer to rougher LER. A label-free per-stack d* (PLAN §4.2 candidate) was not implemented. The secondary was missing from the first run and was added (Exp 1 rerun; same primary numbers).
2. **Near-miss capture hits its target, but read it carefully.** Delta-beta alone catches 29% of notch/filament (filament 17%; recall by gap size 1/2/3 nm: delta-beta 30%/10%/10%, full rule 100% each, n = 23/20/29), so most near-misses read as nominal topology; the energy+bottleneck rule then catches nearly all, at a nuisance alarm of 4% (n = 72). That rule responds to any residual defect energy in a patch whose topology is unchanged; Exp 5 contains no harmless-but-large nuisance variety, so it does not show the rule separates a near-miss from a rough-but-harmless wire (Exp 1 suggests it does not at sigma >= 1.5).
3. **The verifier does not reach the CNN's in-distribution accuracy**, and only beats it zero-shot on via. The honest summary of this testbed: topology-first verification is much better than die-to-die, comparable to or better than a Manhattan-trained CNN off-distribution, and worse in-distribution, with a latency ~40x higher.
4. **Alignment is necessary but not sufficient for Exp 4:** 94% with vs 39% without Module 3. Remaining flips with Module 3 (n = 192): bridge 90%, open 92%, nuisance/particle 98% unchanged; flips are mostly NEAR_MISS <-> KILLER_BRIDGE (5) and open/particle <-> nuisance (4), consistent with the sub-pixel floor measured in M4.

### Rulings
1. **Singleton path is the primary verifier in all experiments;** Module 2 (stack) is only timed (Exp 6), not scored. Reason: M5 showed the stack field does not beat per-patch smoothing for Module 4. Cost if wrong: experiments understate a stack-based verifier on layouts with peers.
2. **Baseline rules are fitted per dose on Manhattan val at the nominal nuisance** (sigma 1.0, grain 15%) and not re-fitted per Exp 1 cell; the CNN threshold (0.39) is from M6. Generous to die-to-die in dose, not in nuisance level.
3. **Config selection uses a fixed small grid on val;** pre-smooth, region_tol and the SDF constants were fixed beforehand, not searched. Chosen on n = 48 val nuisance samples (suppression granularity 2%).
4. **CNN: one training run (seed 0, Colab GPU),** so the CIs reflect test-sample variance only, not training-seed variance. GPU training is not bit-reproducible.
5. **Near-miss definition for Exp 5:** an alarm is any non-NUISANCE verdict (Delta-beta != 0 or NEAR_MISS); strict NEAR_MISS-class rates are also in `results/exp5/near_miss.csv`.

### Caveats
- Exp 1 cells and Exp 5 classes have n = 24-72; many CIs are wide (see README). Exp 4's miss is within its CI of the 95% target.
- Full-run reproducibility was checked by tests on a tiny config (identical tables for Exp 1/3/4/5) and by rerunning Exp 1 (identical primary numbers); the whole full run was not repeated end to end.
- `reproduce.py --quick` (150 s) writes `results/quick/` and leaves the README alone.

### Next
M8: `notebooks/demo.ipynb` (one tile through every stage with each verdict and reason; plan: logic in a tested module, since `jupyter`/`nbconvert` is not an allowed dependency), regenerate the review-UI data with the current pipeline, and a final PROGRESS/README pass.

---

## Milestone M8: packaging (demo notebook, review UI refresh, README)
Status: done. `pytest -q` = 163 passed (6 new demo/export tests). No new dependency (jupyter/nbconvert are not installed or needed).
- **`src/nanovate/demo.py`** (`walk_through`, `describe`, `plot_walk_through`, `load_frozen_cfg`) + **`notebooks/demo.ipynb`** (generated by `scripts/make_demo_notebook.py`, no stored outputs). The notebook is a thin caller; `tests/test_demo.py` executes every code cell headlessly (ruling from PLAN M8: logic in a tested module because `nbconvert` is not an allowed dependency). Illustrative tiles (val split, dose 100): bridge -> KILLER_BRIDGE (delta-beta (-1, 0)); open and particle -> KILLER_OPEN/PARTICLE (1, 0); nuisance -> NUISANCE; notch -> NEAR_MISS. These are single illustrations, not results.
- **Review UI** (`scripts/export_ui.py`): now runs each of 36 samples through the real pipeline (calibrated levels, alignment, SDF energy, frozen config); the old oracle-level / no-SDF export is gone. Export values are rounded; `tests/test_export_ui.py` checks structure, calibrated levels and that every sample carries an SDF energy. Dose is a flag (`--dose`, default 50); the UI shows one dose only.
- **README**: thesis, ASCII pipeline, how to run, demo and UI sections, limitations; the results block is generated by `scripts/reproduce.py` from the CSVs.
- Missed targets (kept as stated; see M7): Exp 1, 2, 4, 6. Met: Exp 3, 5.

### Not done / open
- No pre-registered target was changed. A label-free per-stack d* (PLAN §4.2 candidate) and a near-miss rule that adapts to LER were not implemented (Exp 1 shows the frozen d*/e* do not transfer to rougher LER).
- Results come from one full run and one CNN training run (seed 0); no seed-to-seed variance is reported.
- The stretch items (blind-spot denoiser, persistence-image few-shot classifier, GPU SVD) were not started.

---

## v2: post-hoc fixes after the v1 test results (`docs/V2_FIXES.md`)
Status: done. `pytest -q` = 170 passed (7 new). Full run `python scripts/reproduce.py` (v2, 36 min) writes `results/v2/`;
v1 stays in `results/` (0 of 40 v1 test verdicts change under the new code with `--version v1`). Synthetic data.
**Every fix was designed after seeing v1 test results; all choices on Manhattan val only; no target changed.**

### Changes
1. **Calibration speed (Exp 6, implementation):** ridge from 5x5 moment sums; shift the blurred channels (shift and blur commute). Equivalent outputs (tests).
2. **Splice-seam holes (implementation bug):** `h1_fill="dielectric"` counts holes with dielectric outside the relevance region (image and CAD alike). The CAD-filled splice closed edge pockets into fake holes, i.e. fake KILLER_BRIDGE on nuisance. Val false killers at LER 1.0 / 1.5 / 2.0 nm: 0.08 / 0.15 / 0.27 -> 0.00 / 0.06 / 0.06 (n = 48 each).
3. **Near-miss rule (selection protocol + statistic):** label-free peak-z (largest energy peak vs the patch's other peaks) and thresholds constrained to <= 5% NEAR_MISS in every val cell (LER 0.5-1.5 nm x grain 0-30%) instead of nominal only.
4. **CAD reference cache in Module 4:** the seam fix doubled persistence calls (topology 79 -> 142 ms/patch); caching the CAD-only reference restored 76 ms with identical outputs (0 of 60 verdicts changed).
5. **Exp 2 diagnostic:** verifier on noise-free images (`results/v2/exp2/ceiling.csv`).

### Frozen v2 config (`results/v2/calib/decisions.json`)
Same field / tau as v1 (template, splice, tau 0.35); val F1 0.864 (v1 0.847), val suppression 1.00 (v1 0.917), fake bridges 0 (v1 4);
d* 0.078, e* 2.62, z* 5.58; worst val cell NEAR_MISS rate 4.2% (n = 48 per cell, 9 cells), val near-miss recall 0.67 (v1 0.65).

### Results vs targets (test; README v2 table is generated from the CSVs)
| Exp | v1 | v2 | Met v2 |
|---|---|---|---|
| 1 worst cell (sigma 1.5, grain 30%, n = 24) | supp. 46%, bridge 79% | supp. 88% [75, 100], bridge 79% [62, 96] | NO |
| 2 F1 at dose >= 50 (n = 144) | 0.81-0.85 | 0.81-0.85 | NO |
| 3 F1 drop curvy / via | -3.7 / -7.3 | -3.7 / -7.3 | yes |
| 4 unchanged (n = 192) | 94% | 94% [91, 97] | NO (CI contains 95%) |
| 5 near-miss recall (n = 144) | 92% | 90% [84, 94] at 4% alarm | yes |
| 6 calibration | 79 ms | 43 ms | yes |

### Why the remaining targets are still missed (one line each, as AGENTS.md section 6 asks)
- Exp 1: suppression is 21/24 in each sigma 1.5 cell (CI contains 90%); bridge recall at grain 30% is a dim-bridge limit (dark grain on a narrow bridge, P 0.33-0.58 < 1 - tau); all 5 misses per cell are NEAR_MISS, none NUISANCE. A larger tau trades opens and filaments for it (val: open 0.84 -> 0.44, filament false bridges 5 -> 42/96); rejected.
- Exp 2: not noise. At dose >= 50 notches with 2-3 nm left are called killers 60% of the time, filaments 1-2 nm short 33%; noise-free F1 is 0.84 [0.77, 0.89]. Physics/approach ceiling of a hard threshold at this blur.
- Exp 4: unchanged; ~0.4 px CAD-template alignment floor + sub-resolution killers flipping with phase.

### Rejected on val (evidence in `scripts/v2_probes/`)
Energy normalised by LER estimates (over-corrects at low LER); edge-band masking (no effect); larger tau for grain bridges (above).

### Next (scope decisions, not started)
Model-based local hypothesis test (cut vs neck) on the Module-1 template; consensus-based alignment for Exp 4; seed variance (3 seeds + second CNN run);
real-data path: independent simulator (Nebula) on the existing testbed wafers, then Carinthia-S for particle/noise checks, then a line/space set with GDS from a partner.

---

## Part 1: the day-0 value case (`docs/superpowers/specs/2026-10-18-next-steps-design.md`)
Branch `day0-value-case`. `pytest -q` = 184 passed (14 new). Synthetic data.

### Built
- `baselines/breakeven.py` (subsets nested in k, threshold labels counted in k, fixed 300-step fine-tune schedule), warm-start `train_cnn(init=)`,
  family-aware `make_cnn_train_set(family=)`; `scripts/breakeven_train.py` + `notebooks/breakeven_colab.ipynb` (Colab, not yet run);
  `experiments/breakeven.py` (local evaluation; not yet run: needs the Colab weights).
- `experiments/day0_scenario.py`: review-queue **projection** from the v2 Exp 3 via test rates (dose 50, n = 24 per kind), site mix assumed
  (killers equal thirds; harmless 90% nuisance / 5% notch / 5% filament); `results/v2/day0_scenario/`.
- `report.render_day0` -> generated blocks in README ("Why use this: the day-0 case") and `docs/USE_CASE.md`.

### Measured (projection, 1,000 sites, 5% killers; numbers generated in README / USE_CASE.md)
The verifier misses the fewest killers on a new via layer (2 [0, 4] vs CNN 12 [9, 16], die-to-die 19-24) and flags the fewest non-killer
sites (incl. near-misses) as killers (22 [12, 34] vs 46-103), but its total queue is the largest (179 [142, 250] vs 71-133) because it also sends a NEAR_MISS
bin: 73 real near-misses plus 36 [0, 107] harmless sites. Whether that bin is a feature (reliability screening) or cost depends on the fab;
both are shown. The wide CI on harmless NEAR_MISS comes from n = 24 nuisance samples.

### Label break-even (run on Kaggle GPU, private kernel devariwala27/nanovate-breakeven, commit 3787540; evaluated locally)
42 fine-tunes (2 families x 7 k x 3 seeds, 300 steps each) of `m6_cnn/cnn.pt`; test n = 144 per family (dose 50), same patches as Exp 3.
Colab was replaced by Kaggle on the user's request (ledger ruling); GPU training is not bit-reproducible, the 3 seeds show the spread.
| Family | Verifier F1 (no labels) | CNN k = 0 | Break-even k (seed-mean CNN F1 >= verifier) | Clearly better k (every seed's CI above) |
|---|---|---|---|---|
| via | 0.88 | 0.74 | 192 (both threshold rules) | none up to 768 |
| curvy | 0.84 | 0.85 | 0 (already tied) | 48 (fixed threshold) / 192 (labelled threshold) |
Generated tables: README "Why use this" and `docs/USE_CASE.md`; data `results/v2/breakeven/`.

Findings:
- **On via, the verifier is worth about 192 labelled images of the new layout**; even with 768 labels the fine-tuned CNN (0.91-0.92) is not clearly
  better than the verifier (0.88 [0.82, 0.93]).
- **On curvy there is no day-0 advantage**: the Manhattan CNN already ties the verifier with zero curvy labels, and pulls clearly ahead with 48-192.
  Reported as is; the day-0 claim holds for the via layout, not for every new layout.
- **Few labels make the CNN worse before better**: at k = 12 the fine-tuned CNN drops below its own k = 0 F1 (via 0.60-0.69 vs 0.74), and the seed
  spread is large (via labelled-threshold 0.38-0.76). A day-0 user with a handful of labels would be worse off fine-tuning.

---

## v3 Phase 0 audit + Phase 1: ceiling diagnosis (`docs/NEXT_PATHS.md`, `docs/V3_PHASE1_DIAGNOSIS.md`)
Status: Phase 1 done, waiting at Stop Point 1. Manhattan **val only**; test split not generated. Synthetic data.
v1 / v2 results, CSVs and README untouched (`git diff` on README.md and results/ is empty).

### Phase 0 audit
- Not done: v2 release tag (no git tags exist); Exp 3 wording (README v2 table still says "via: -7.3 pts"); demo script
  with a CNN-miss case, a near-miss-only case and a failure case (no such curated list in the repo).
- Done: demo app (commits cedd59c..9c51bf1).
- Outside the repo (not attempted): GitHub "About" tagline, creating the release, PPT slide 5 (deck not in the repo).
- Conflict, not resolved: the Exp 3 wording fix changes the generated v2 README block, which the v3 brief requires to stay
  byte-identical. Left unchanged pending a decision.

### Built
`testbed/resample.py` (same wafer at s x s sub-pixels per nm, optics fixed in nm; bit-identical at s = 1),
`diagnostics.py` (instrumented frozen v2 verifier with px-defined parameters rescaled, Delta-beta(tau) curve, defect
saddle values), `experiments/v3_phase1.py`, `scripts/v3_phase1_report.py`. 17 new tests (`tests/test_v3_diagnostics.py`).

### Measured (val, n = 288, 48 per kind, noise-free, dose 50)
- Noise-free killer F1 at 1x / 2x / 3x sampling: 0.85 [0.81, 0.89] / 0.85 [0.81, 0.89] / 0.84 [0.80, 0.89]. Flat.
- Supplementary, beam + bloom halved in nm (2x sampling): 0.92 [0.88, 0.95].
- 46 wrong killer flags at 1x: 13 missed killers (bridge 5, open 8) and 33 false killers (notch 23, filament 8, nuisance 2).
  45/46 are recoverable by a per-sample oracle tau; the best single in-sample tau gives only F1 0.87.
- Separability of the defect saddle (fewest errors of ANY threshold): bridge vs filament 3/96 (AUC 0.995);
  open vs notch 25/95 (AUC 0.82) at 1x, 2x and 3x alike; 4/95 with halved blur.

### Surprises
1. The noise-free ceiling is driven more by **false killers** (33, mostly 2-3 nm-remaining notches called opens) than by
   missed killers (13). NEXT_PATHS frames it as "misses".
2. Two of five bridge "misses" are not missed at the bridge: the merge is seen, but a counted feature elsewhere in the
   patch cancels it in Delta-beta. Six filaments read as a detached particle (thin base dims), not as a bridge.
3. The decision rule in NEXT_PATHS ("F1 flat across sampling -> logic-limited") cannot detect an optical limit,
   because sampling at fixed blur in nm does not change resolution. The oracle-tau test is also weak: with one tau per
   patch nearly everything is recoverable. Both literal signals say "logic-limited"; the cut class is not.
4. The v2 README sentence "killers missed even noise-free (below this verifier's optical resolution)" is wrong for
   bridges. Not edited (v2 byte-identical); corrected in V3_PHASE1_DIAGNOSIS.md.

### Next
Stop Point 1: decision on Phase 2 scope (see the diagnosis conclusion).

### Phase 2 design (draft, `docs/V3_PHASE2_DESIGN.md`), waiting at Stop Point 2
Val-only design probe (`scripts/v3_probes/design_probe.py`, noise-free, n = 96 bridge+filament per grain level):
bridge vs filament merge saddle, fewest errors of any threshold: grain 15% 3/96 (AUC 0.995), grain 30% 7/96 (AUC 0.983);
normalising by the defect body level: 4/96 and 5/96. Grain-dimmed bridges are mostly a threshold-placement problem
noise-free; Exp 1 is noisy, so this is an optimistic bound. The filament-read-as-split cue was not measured (probe gap).

### Phase 2 implementation + val selection (`docs/V3_PHASE2_VAL.md`), waiting at Stop Point 3
Built `modules/events.py` (located events: per-seed-pair merge level vs the template-expected level, H0/H1 bars located
by GUDHI vertex pairs, event-set verdict), `pipeline_v3.py` (v2 path untouched; `rule="v2_global_tau"` calls
`verify_patch`), `experiments/v3_select.py`, `experiments/v3_val_report.py`, v3 branch in `experiments/report.py`
(Exp 3 reported as change, "+7.3 pts (improved)", v3 block only; v2 block still renders byte-identically, tested).
15 new tests (`tests/test_v3_events.py`); v2 Delta-beta and bottleneck reproduced from the located bars.

Frozen on Manhattan val (`results/v3/calib/decisions.json`, 2,592 patches): m_merge 0.60, normalisation off,
per-neck m_cut 0.40, e_edge off; d* 0.078, e* 2.57, z* 5.46. Pooled val F1 0.90 [0.88, 0.91] vs v2 0.85 [0.83, 0.86].

Surprises / honest notes:
1. The per-neck cut margin was selected and gives most of the pooled gain (without it the best pooled F1 is 0.855).
   It does not separate cuts from necks better. It moves the operating point: notch called killer 43% -> 9%,
   open recall 85% -> 74% (in-selection, n = 432 each). v3 therefore lets more opens escape than v2.
2. The protrusion cue (e_edge) lowered val F1 at every setting, so it is off: filament-as-split is not fixed (16-28% of
   filaments still called killer).
3. Held-out val (layouts 8-23, dose 50): noisy F1 0.82 -> 0.85, noise-free 0.85 -> 0.88 (n = 192; paired differences'
   CIs include 0). Noise-free target 0.92 looks likely to be missed; Exp 1 val analogue worst cell bridge recall 81%
   (sigma 1.5, grain 30%, n = 16), up from 62%.
4. Cut detection limit (pre-registered definition, noise-free val): 5 nm cut length.
5. Pre-existing drift found: the current generator renders the v1 README header as "(full run, v1; ...)" while the
   README v1 block says "(full run; ...)"; `reproduce.py --version v1` would change that line. Not touched.

### v3 fresh-test run (once; test seed 20261018; README "Results (v3)" generated from `results/v3/`)
Ran `python -m experiments.v3_run` once (26 min). Primary = frozen config; secondary (cut margin off) pre-declared in
`results/v3/calib/secondary.json` (commit c8728bf) before the run; "v2 rule" re-scored on the same patches (paired).
Disclosure: a first `--quick` smoke run accidentally drew 3 layouts x 1 die from the official test seed; its output was
deleted unread, nothing was changed afterwards, and `--quick` now uses a separate smoke seed (test_seed + 7919, tested).

Phase 2 targets (all missed; kept as stated):
| Target | Measured (v3 primary) | Met |
|---|---|---|
| Noise-free F1 >= 0.92 | 0.89 [0.83, 0.94] (n = 144) | NO: cut-class optical limit (4 of 24 opens are 3-4 nm cuts, all missed noise-free) |
| Exp 1 suppression >= 90% | worst cell 67% [50, 83] (sigma 1.5, grain 20%, n = 24) | NO: same as the v2 rule on these patches (67%); spurious killer events at high LER |
| Exp 1 bridge recall >= 90% | worst cell 88% [75, 100] (sigma 0.5, grain 30%, n = 24) | NO, but CI contains 90%; v2 rule on the same patches 62% |
| Manhattan F1 >= 0.88 | 0.87 [0.81, 0.92] (n = 144) | NO (CI contains it); v2 rule same patches 0.86 |
Original targets: Exp 1 NO, Exp 2 NO (0.87-0.89 at dose >= 50), Exp 3 yes (curvy +2.9, via -2.6 pts), Exp 4 yes (97%; v2 test 94%),
Exp 5 yes (85% recall), Exp 6 yes (43 ms). Exp 5 note: test nuisance alarm is 10% [4, 17] (n = 72), above the 5% val budget,
all from killer events (Delta-beta alone gives the same 10%), so the "fixed false-alarm rate" is not held on this seed.

Paired v3 vs v2 rule on the fresh patches: bridge worst-cell recall 62% -> 88%, notch called killer 33% -> 12%, Manhattan F1
0.86 -> 0.87, noise-free 0.87 -> 0.89, but open recall 88% -> 83% (the cut margin's cost; secondary keeps 88%) and filaments
called killer 29% -> 38%. Per-patch time 99 -> 130 ms (Module 4 76 -> 107 ms). The CNN still leads in-domain (0.92) and
the verifier leads zero-shot on via (0.84 vs 0.75). Cut detection limit on test: 4 nm (n = 4-7 per bin; val said 5 nm).
Surprise: the fresh test seed is harder for nuisance at sigma 1.5 than v2's test seed (v2 rule 67% here vs 88% there), so
cross-seed comparison with the v2 README table is not like-for-like; use the paired table.

## Track C Rung 1: real SEM images (pre-registration, before the first real run)

- `docs/NEXT_PATHS.md`: Track C (real-world data, Rungs 1-4) added and first in the suggested order (commit de73db3).
- Data: Zenodo 8060099 (ZMW hole arrays, CC-BY-4.0) and 3747155 (InGaN figure panels, spot-check), md5-checked by
  `scripts/real/download_rung1.py` into `data/real/` (git-ignored). Manifest + exclusions in `results/real/rung1/`.
- Surprise 1: the approved pitch-matching rescale is infeasible (pitch/diameter 5-8 vs testbed 2.25: a G29 image would
  shrink to ~99 x 66 px). Arm C (native px) is primary, arm B (diameter-matched) secondary; decided with the user before the run.
- Surprise 2: every G29 TIFF has a burned-in yellow centre cross (61 px); flagged from R != B, never inpainted; overlay
  tiles leave the primary denominator but are still reported.
- Surprise 3: the G01 arrays are diameter series (hole size changes by column), so one fitted diameter mismatches many
  holes (size, not topology). The InGaN panels show small pits between holes, so they are labelled not clean.
- Clean labels are a visual judgement by the assistant (not an expert), made before any verifier run.
- Pre-registration: `docs/REAL_DATA_PREREG.md` (target: KILLER on clean real tiles <= 5%).
