# PROGRESS.md — Nanovate SEM Topology Defect Verifier

Status: **M1 complete. M2 code + tests green (45 passed); M2 early-risk probe and field/tau choice still open.**
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

## Milestone M6 (in progress): baselines (`baselines/die_to_die.py`, `baselines/cnn.py`, `scripts/baselines_report.py`)
Status: code + tests done (11 new tests pass: D2D flags an obvious bridge, not a clean die or re-acquisition; double-difference cancels a defect in one reference; val-only threshold fit refuses other splits; CNN forward shape, seeded determinism, tiny-batch overfit, val-only threshold). **Baseline numbers are NOT final.**
D2D aligns target and references with the Module-3 aligner and applies the Module-1 GAT (z-units), so alignment and noise scaling are not confounds.

First full run on Manhattan val (synthetic; 50% killer prevalence, so "flag everything" scores F1 = 0.67), both baselines degenerate:
- **D2D with one pooled rule over doses 20/50/200**: F1 0.68, nuisance false alarm 0.71 overall; at dose 50 and 200 it flags every clean die. LER differences between dies grow relative to shot noise as dose rises, so a z-threshold cannot transfer across doses. Fix in the script (not yet rerun): fit the rule per dose (recipe tuned per condition, the best case for D2D), keep the pooled rule as a comparison.
- **CNN**: training loss 0.733 -> 0.701 (chance), i.e. untrained: 10 epochs x 18 steps on 576 mixed-dose samples is too few steps to judge. A longer schedule is to be trained off this machine (Colab); until then no CNN number is reported.

### M6: CNN training moved off this machine (Colab)
`scripts/train_cnn.py` + `notebooks/train_cnn_colab.ipynb` train the CNN on a GPU: 48 Manhattan-train layouts x 4 dies x 6 kinds = 1152 patches, random dose per sample (5-200 e-/px), 60 epochs (~2,200 steps), seed 0; the budget is fixed before any val/test CNN number exists. Weights come back as `cnn.pt` and are evaluated here with `scripts/baselines_report.py --cnn-weights` (val-chosen threshold, F1, latency). GPU training is not bit-reproducible; config, seed and loss history are saved with the weights. Not yet run.
