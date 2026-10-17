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
