# Nanovate demonstrator app — design spec (2026-10-18)

Status: **approved 2026-10-18** with the defaults in §11 (httpx added to `[app]`; seed = new wafer, frozen methods; dose grid only; non-256 tiles refused; on-disk D2D rule cache).

A local web app (localhost only) where a user generates or loads one SEM patch, runs the verifier and both
baselines on it, and sees each verdict with its reason, its ground truth when one exists, and its provenance.
It is a thin layer over the research code. It does not scan wafers, control a tool, or add detection logic.
All data it can produce is synthetic, except files the user loads, and those have no ground truth.

## 1. What I read and what it implies

| Finding | Consequence for the app |
|---|---|
| `experiments/harness.py::Context` is what every experiment uses: frozen `PipelineCfg` (v2), `calibrate(make_scribe_patches(tcfg, seed))` cached per (dose, LER, grain), D2D references = nuisance dies 900/901 of the same layout, **D2D rules fitted per dose on Manhattan val at runtime**, CNN = `m6_cnn/cnn.pt` with threshold from `results/m6_baselines/rules.json`. | The app **reuses `Context` objects and the same core calls**, not copies of them. `Context.evaluate` is the parity oracle in tests. |
| D2D rule fit for one dose takes **~21 s** on this CPU (measured today: 96 val samples + refs). A rule exists only for a dose it was fitted at. | Dose is a **select from the Exp-2 grid** (5, 10, 20, 35, 50, 100, 200 e⁻/px), not a free slider. First use of a dose fits its rule once (UI says so), then it is cached in memory and on disk (§6). |
| One `Context.evaluate` (verifier + D2D + CNN, cached calibration) took 0.27–0.6 s. | `/verify` is synchronous; expected p50 well under 1 s after warm-up. |
| `verify_patch`'s reason has Δβ, β, bottleneck H0/H1/max, SDF energy, peak-z, thresholds — but **no location**. | The "region driving Δβ" marker needs a display-only locator (§7). It never feeds a verdict and is labelled as such. |
| `demo.walk_through` rebuilds the conductor field and energy map with the same calls the pipeline uses. | The app returns the field and energy map the same way, and a test checks they equal `walk_through`'s for the same sample. |
| `experiments/report.py::render_tables(root)` and `render_day0(root)` generate the README blocks from the CSVs. | `/results` calls these two functions and parses their markdown tables into JSON. No number is formatted by the app. |
| `frontend/index.html`: dark IDE-style layout (title/menu/tool bars, left Explorer + Patch Layers, centre viewport with tabs, right Inspector/Metrics, bottom Sample List/Pipeline/Log, status bar), CSS variables `--bg-*`, `--killer`, `--near-miss`, `--nuisance`, `--ok`, `--warn`, `--danger`. | All new UI goes into these existing panels/tabs and uses only these variables. |
| Stale text in `index.html`: Pipeline banner ("Modules 1–3 are not built"), Inspector "n/a (Module 3 not built)", log lines "Module 4 only" and "untuned Module 4", Regenerate modal shows flags `export_ui.py` no longer accepts (`--tau --field --pre-smooth`). | Fixed in M3 (wording only). |
| fastapi/uvicorn/pydantic/multipart/tifffile are not installed; pillow is (used by `export_ui.py`). `pytest -q` is green today. | New deps go in an optional `[app]` extra; app tests `importorskip("fastapi")` so the research install stays green without it. |

## 2. Architecture

```
frontend/index.html (vanilla JS)  ──fetch──►  FastAPI  /api/v1/*   (src/nanovate/app/api.py)
   static grid from data/run.js                    │ pydantic in/out, request-id, size cap, error handlers
   Live / Upload / Evidence                        ▼
                                         Engine (src/nanovate/app/engine.py)        one lock: one analysis at a time
                                           ├─ InputStore   id -> InspectionInput (in-memory LRU, 64 items)
                                           ├─ CalibrationCache  layer_id -> (Calibration, ms)
                                           ├─ harness.Context(ExpCfg.full())  -> frozen PipelineCfg, CNN + thr, D2D rules
                                           └─ calls: verify_patch · d2d_score/d2d_flag · predict_proba · energy_map/build_field
                                         Sources (src/nanovate/app/sources.py)
                                           SimulatedSource · FileSource · ToolSource (stub)
```

Package `src/nanovate/app/`:

| File | Content |
|---|---|
| `__main__.py` | `python -m nanovate.app [--port 8765] [--host 127.0.0.1]`. Refuses a non-loopback host. Checks it runs from the repo root (needs `results/`, `m6_cnn/`, `experiments/`) and exits with a clear message otherwise. |
| `sources.py` | `InspectionInput`, `ImageSource` protocol, the three sources. |
| `engine.py` | Context wiring, caches, `analyse()`, overlays. |
| `explain.py` | Template explanations and the display-only Δβ locator. |
| `evidence.py` | `/results` parsing of `render_tables` / `render_day0` output. |
| `report.py` | JSON + HTML report of one analysis. |
| `schemas.py`, `api.py` | pydantic models, routes, errors. |

`experiments/` is a top-level package; the app imports `experiments.harness` (as the tests already do) so there is
exactly one implementation of "how an experiment runs a sample". This makes the app depend on running from the
repo root, which it needs anyway for the weights and CSVs.

**Frozen artefacts (never changed by the app):** `PipelineCfg` from `results/v2/calib/decisions.json` (via
`load_pipeline_cfg(ExpCfg.full())`), `m6_cnn/cnn.pt`, `cnn_threshold` from `results/m6_baselines/rules.json`,
D2D rules from `Context.d2d_rules(dose)` (seed 0, Manhattan val, same procedure as the v2 results).
**Config hash** = sha256 of canonical JSON {pipeline_cfg, TestbedCfg defaults, cnn_threshold, sha256(cnn.pt),
ExpCfg.full() fields used by the rule fit}. **Commit** = `git rev-parse HEAD` at start (+ `-dirty` if the tree is dirty).

## 3. Input-source interface

```python
@dataclass(frozen=True)
class InputMeta:
    pixel_nm: float            # nm/px (verifier built for 1.0)
    dose: float | None         # e-/px, None if unknown
    layer_id: str              # calibration cache key
    source_type: str           # 'simulated' | 'file' | 'tool'
    provenance: str            # exact badge text, see below
    params: dict               # source parameters (simulation params, file names + sha256)

@dataclass(frozen=True)
class GroundTruth:             # only from the testbed's ground-truth wafer W
    kind: str; label: str      # injected kind; NUISANCE / NEAR_MISS / KILLER (from W vs CAD)
    d_beta: tuple[int, int]; W: np.ndarray; mask: np.ndarray; expected: np.ndarray; resamples: int

@dataclass(frozen=True)
class CalibrationRef:          # exactly one of the two
    scribe: list[tuple[np.ndarray, np.ndarray]] | None   # [(cad, image)] x 3
    calibration_id: str | None                           # an id already in CalibrationCache

@dataclass(frozen=True)
class InspectionInput:
    image: np.ndarray          # float32 (H, W), e-/px
    cad: np.ndarray            # bool (H, W)
    meta: InputMeta
    calibration: CalibrationRef
    truth: GroundTruth | None
    references: list[np.ndarray]   # 0-2 reference dies for die-to-die (same CAD frame)
    warnings: list[str]            # out-of-domain warnings, shown prominently

class ImageSource(Protocol):
    def load(self) -> InspectionInput: ...
```

- **`SimulatedSource(family, kind, dose, ler_sigma, grain, shift: (dy, dx) | None, warp, seed, split='live', layout_id=0, die=0)`**
  builds `tcfg = nuisance_cfg(with_dose(TestbedCfg(), dose), ler_sigma, grain)` (the harness helpers) and calls
  `get_layout` / `make_sample` / `make_scribe_patches(tcfg, seed)`; references = `make_sample(..., die=900+j, 'nuisance')`
  exactly as `Context.refs`. Ranges: family ∈ {manhattan, curvy, via}; kind ∈ `KINDS`; dose ∈ Exp-2 grid; LER σ
  0.5–2.5 nm; grain 0–0.30; shift each axis ±8 px or `None` (drawn from the acquisition stream as in the experiments);
  warp 0–1 px; seed 0–2³¹−1. `layer_id = f"sim/{dose}/{ler}/{grain}/seed{seed}"`.
  - `split='live'` (default for Live mode): layouts under a split key never used by any experiment, badge
    **"SIMULATED (Nanovate testbed)"**.
  - `split ∈ {'val','test'}`: the exact experiment sample; badge **"SIMULATED (val/test split sample)"** (with the
    split named). `train` is refused (the CNN saw it).
  - The static grid (`run.js`) keeps its own badge: "SIMULATED (val/test split sample)".
- **`FileSource(image_file, cad_file, sidecar_json, scribe_files | calibration_id, reference_files=[])`**
  - image: PNG (8/16-bit gray) or TIFF (via `tifffile`; float32 allowed). Values are taken as e⁻/px counts; an
    8/16-bit PNG requires `"intensity_scale"` in the sidecar (e⁻/px per grey level) or it is refused — no guessing.
  - CAD: PNG mask (any non-zero = conductor), or GDS via `gdstk` + the testbed's `rasterize` when the sidecar gives
    `cell`, `layer`, `datatype` and `origin_nm` (only if this stays under ~40 lines; otherwise PNG only, and I'll say so).
  - sidecar schema (pydantic): `pixel_nm` (required), `layer_id` (required), `dose` (optional), `intensity_scale`
    (PNG only), optional `notes`. Unknown keys are rejected.
  - calibration: 3 scribe pairs (CAD PNG + image) **or** a `calibration_id` present in the cache. Neither → 422
    `calibration_missing` with the message "Upload 3 scribe patches or name an existing calibration id; the verifier
    will not guess the conductor/dielectric levels."
  - domain: `pixel_nm != 1.0` → **refused** (422 `pixel_size_out_of_domain`: every length in the verifier is in px
    at 1 nm/px). Tile ≠ 256×256 → **refused** for now (window, margin and the CNN input assume 256; a stated
    limitation, not silently resized). Never infers nm/px from the image size.
  - die-to-die needs 1 reference image (single) and 2 (double); with none, the D2D card says "not available: no
    reference die uploaded" rather than inventing one. D2D also needs `dose` in the Exp-2 grid for a rule; otherwise
    "not available: no rule fitted at this dose".
  - CNN runs, but the card carries "trained on simulated Manhattan; outside its training domain" for every file input.
  - badge **"LOADED FROM FILE"**; `truth = None` always.
- **`ToolSource`**: a `Protocol` subclass with `load()` raising `NotImplementedError`, and a docstring listing what a
  vendor adapter must deliver: exported review images with pixel size and dose per image; site coordinates (KLARF
  `DefectList` X/Y + die index) mapped to the CAD frame; the CAD clip per site (GDS/OASIS window); scribe images per
  layer; and the acquisition recipe id as `layer_id`. No implementation, no fake data, no route.

## 4. API contract (`/api/v1`, JSON, pydantic)

Every response carries `request_id`, `api_version`, `commit`, `config_hash`. Every analysis response echoes `seed`.
Images are PNG data URIs (8-bit, deterministic encoding) plus their display range; arrays never leave as floats
except in the report and, with `include_arrays=true`, as base64 float32 (used by tests).

| Route | Request | Response |
|---|---|---|
| `GET /health` | – | `version`, `commit`, `config_hash`, `cnn_loaded`, `cnn_threshold`, `frozen_cfg` (dict), `d2d_rules_cached` (doses), `calibrations_cached`. |
| `POST /simulate` | SimulatedSource params | `input_id`, `provenance`, `seed`, images `{cad, wafer, sem, expected}`, `truth {kind, label, d_beta, resamples}`, `applied_shift`, `params`. |
| `POST /verify` | `{input_id}` **or** `{simulate: {...}}` (inline) | `verifier {verdict, reason{delta_betti, beta_ref, beta_img, bottleneck_h0, bottleneck_h1, bottleneck_max, max_sdf_energy, peak_z, tau, d_star, e_star, z_star, singleton}, recovered_shift, timings_ms{align, sdf_energy, topology}}`, `cnn {prob, threshold, flagged, ms, domain_note}`, `d2d {single, double: {flagged, max_area, rule} | {available:false, why}, ms}`, `truth` or `null`, `agreement {verifier, cnn, d2d: 'agrees'|'disagrees'|'no ground truth'}`, `overlays {field, sdf_energy, cad_outline, locator}`, `explanation [sentences]`, `calibration {layer_id, ms, cached}`, `seed`, `provenance`, `warnings`. |
| `POST /upload` | multipart: `image`, `cad`, `sidecar`, `scribe_cad[3]`+`scribe_image[3]` or `calibration_id`, `reference[0..2]` | `input_id`, `provenance`, `warnings`, images; or a structured 4xx (below). |
| `GET /results` | – | `{readme: {markdown, tables:[{title, headers, rows}]}, day0: {...}, sources: [csv paths read], cnn_speed_ratio: {value, from: 'results/v2/exp6/latency.csv', rows used}}`. |
| `POST /report` | `{input_id}` (re-runs `/verify`; deterministic) and `format: 'json'|'html'` | download (`Content-Disposition`) with inputs (PNG + sha256), provenance, config, config hash, seed, commit, all three verdicts, reasons, timings, the "synthetic data" / "loaded from file, no ground truth" label. |

Agreement for D2D/CNN uses killer vs not-killer (they have no NEAR_MISS bin); for the verifier, exact label match
with KILLER_* → KILLER. Without ground truth the cards show "no ground truth" and **no** green or red mark.

**Errors.** Body `{error: {code, message, field?, request_id}}`.
400 `bad_request` · 413 `too_large` (request > 20 MB, any file > 8 MB, any side > 2048 px) · 415 `unsupported_format` ·
422 codes: `shape_mismatch`, `bad_dtype`, `non_finite`, `negative_counts`, `empty_cad`, `full_cad`, `bad_sidecar`,
`pixel_size_out_of_domain`, `tile_size_out_of_domain`, `calibration_missing`, `unknown_calibration_id`,
`out_of_range` (simulate params), `split_not_allowed` · 404 `unknown_input_id` · 500 `internal` (message: "internal
error, request id X" — the traceback is logged server-side with the request id, never sent). Exception handlers
convert everything; one bad request never takes the server down (tested by sending a bad request, then a good one).

**Determinism.** `seed` drives the testbed sample, the scribe patches and the D2D reference dies. The frozen
artefacts (verifier config, CNN weights/threshold, D2D rules) are fixed from the experiments (seed 0) and do not
change with the request seed — a request seed is a different wafer, not a different method. Torch runs with
`torch.use_deterministic_algorithms(True)` semantics already given by eval mode on CPU. `input_id` =
sha256 of canonical params (simulated) or file bytes + sidecar (file), so the same request yields the same id.
Same request → identical response except the `timings_ms`, `ms` and `request_id` fields.

## 5. Frontend (vanilla JS, existing panels and CSS variables only)

- **Mode tabs** in the viewport header: `Sample Grid` (renamed label: "validation split, synthetic"), `Patch Compare`,
  new `Live`, new `Evidence`.
- **Live controls**: shown in the left column in place of Run Explorer while `Live` is active (same `.panel` /
  `.panel-header` styling): family, kind, dose (select), LER σ, grain, shift dy/dx (+ "random" checkbox), warp,
  seed, split (live / val / test), **Randomise** (client draws explicit values, so the request stays reproducible),
  **Run**, **Cancel** (AbortController; the server finishes the job and discards it — stated in the tooltip).
  Loading state: spinner + "fitting die-to-die rule for dose X on Manhattan val (one time, ~20 s)" when that applies.
- **Patch view** (Live): SEM image with the provenance badge pinned top-left (`.sim-badge` style; amber `--warn`
  for LOADED FROM FILE). Overlay toggles live in the existing **Patch Layers** panel: SEM, CAD outline, conductor
  field, SDF energy heatmap, Δβ locator marker, ground-truth wafer W (only if truth exists). Selecting a reason row
  (Δβ₀ / Δβ₁ / bottleneck / SDF energy) highlights its location (locator box or energy peak).
- **Verdict cards**: three side-by-side cards at the top of the right column (new `Verdicts` tab, default in Live):
  Verifier / CNN / Die-to-die, each with verdict (colour from `--killer` / `--near-miss` / `--nuisance`), ground
  truth or "no ground truth", agreement, its ms. The Inspector below shows the full reason table, seed, config hash,
  commit, calibration id + ms, warnings.
- **Explanation** box under the cards: template sentences (§7).
- **Evidence tab**: the README result tables and day-0 tables rendered as `.data-table`, each with its CSV path(s),
  the "synthetic data" header line, and a fixed callout listing the losses (verifier loses on Manhattan; CNN ×N
  faster with N from `cnn_speed_ratio`; curvy shows no day-0 advantage — break-even k = 0). The callout text is a
  template whose numbers all come from `/results`.
- **Upload panel**: a modal (existing `.modal` style) with file inputs and sidecar fields; validation errors shown
  inline next to the offending field from `error.field`.
- **Error states**: banner in the viewport for "server not reachable — static grid still works; start with
  `python -m nanovate.app`" (also the file:// case); inline upload errors; "calibration missing" with what to upload;
  out-of-domain warning as a `--warn` banner above the patch view, not a toast.
- **Stale text fixes**: Pipeline banner → states Modules 1, 3, 4 run per patch, Module 2 only on stacks (off in the
  frozen config, `use_stack=false`); Inspector "n/a (Module 3 not built)" removed; the two log lines; the
  Regenerate modal command updated to the real `export_ui.py` flags.

## 6. Caches and performance

- `CalibrationCache`: layer_id → (Calibration, fit ms, scribe sha256). Report `{layer_id, ms, cached}` on every verify.
  In-memory; ids live for the server's lifetime.
- D2D rules: `Context.d2d_rules(dose)` in memory, plus a JSON cache `results/app_cache/d2d_rules.json` keyed by
  config hash + dose, written after the first fit (`D2DRule` is a frozen dataclass of four numbers; reloading
  equals refitting — tested). `results/app_cache/` is git-ignored. A `--warm 50` start flag pre-fits dose 50.
- One `threading.Lock` around analysis (GUDHI/torch thread-safety and honest per-module timings); routes run the
  work in FastAPI's threadpool so `/health` stays responsive.
- Latency: `scripts/app_latency.py` runs 50 `/verify` calls through TestClient (warm caches, fixed seeds) and
  prints p50/p95; numbers go into PROGRESS.md. The research core is not optimised.

## 7. Explanations and the Δβ locator

- **Explanations** (`explain.py`): pure functions of the reason dict, one sentence per fact, e.g.
  Δβ₀ = −1 → "β₀ dropped by 1: two separate CAD conductors are connected in the image → bridge (KILLER_BRIDGE)";
  Δβ₀ = +1 → "β₀ rose by 1: …split, or an isolated conductive blob → open/particle"; Δβ₁ ± the hole versions;
  Δβ = 0 + near-miss rule → "topology matches CAD, but bottleneck H₀ 0.09 ≥ d* 0.078, SDF energy 3.1 ≥ e* 2.6 and
  peak-z 6.0 ≥ z* 5.6 → NEAR_MISS"; otherwise which condition failed → NUISANCE. Singleton flag and recovered shift
  get one sentence each. CNN: "probability 0.83 ≥ threshold 0.39 (chosen on Manhattan val); gives no reason".
  D2D: "largest region above θ covers N px ≥ min area M". Numbers are formatted from the response, never typed.
- **Locator (display only)**: binarise the returned field at 0.5 inside the comparison window, label connected
  components of field and CAD; for Δβ₀ < 0 mark image components overlapping ≥ 2 CAD components (bbox of the
  overlap of the image component outside the CAD), for Δβ₀ > 0 mark CAD components covered by ≥ 2 image components
  or image components with no CAD overlap; for near-miss/nuisance mark the SDF-energy argmax. Returned as
  `{boxes, method: 'display-only heuristic; not used by any verdict', agrees_with_delta_beta: bool}`. When it cannot
  find a site consistent with Δβ it says so and shows nothing rather than a guess.

## 8. Dependencies

`pyproject.toml`: `[project.optional-dependencies] app = ["fastapi", "uvicorn", "pydantic", "python-multipart",
"pillow", "tifffile"]`. `requirements-app.txt` pins the exact versions I install (and the research deps already in
`.venv`). Nothing else; no frontend build tools. `httpx` is needed by FastAPI's `TestClient` — **this is outside the
approved list; I'd like approval to add it to the `[app]` extra** (test-only). Alternative without it: test the
routes by calling the handler functions directly, which skips the HTTP layer (size caps, multipart, status codes).

## 9. Milestones (one commit each, `pytest -q` green before each)

1. **Spec approved** (this file).
2. **Sources + API + parity tests**: `sources.py`, `engine.py`, `schemas.py`, `api.py`, `__main__.py`, `/health`,
   `/simulate`, `/verify`; `[app]` extra + pinned requirements; parity, determinism, smoke and validation tests for
   these routes.
3. **Live mode + verdict cards**: Live tab, controls, loading/cancel, three cards, provenance badges, server-down
   state, stale-text fixes.
4. **Overlays + explanations**: field / SDF energy / CAD outline / W overlays, locator, reason→location highlight,
   `explain.py` + its tests.
5. **Upload + Evidence + report**: `FileSource`, `/upload` + all validation errors, `/results` + Evidence tab,
   `/report` (JSON + HTML), round-trip test.
6. **README + PROGRESS**: "Run the app" section, latency p50/p95 from `scripts/app_latency.py`, surprises.

## 10. Test plan (`tests/app/`, each file starts with `pytest.importorskip("fastapi")`)

| Test | What it asserts |
|---|---|
| `test_parity.py` | For seeds {0, 1, 7} × kinds {nuisance, bridge, open, particle, notch, filament} × family manhattan (+ one via, one curvy): `/verify` verifier verdict + every reason field == `verify_patch(...)` called directly on the same sample, cal, window and frozen cfg; `cnn.prob` == `predict_proba(...)` and flag == `prob >= thr`; D2D flags/areas == `d2d_flag(d2d_score(...))`. For seed 0, split test: the response equals `Context(ExpCfg.full()).evaluate(...)` row field by field (the experiments' own path). Field and energy map == `demo.walk_through` for one sample. |
| `test_determinism.py` | Same `/verify` request twice → identical JSON after dropping timing keys and `request_id`; same for `/simulate` images (byte-equal PNGs) and `/report` JSON. Seed is echoed. |
| `test_validation.py` | wrong shape, wrong dtype (e.g. RGB or complex TIFF), NaN, negative counts, empty CAD, mismatched CAD/image sizes, missing calibration, unknown calibration id, oversized file (> cap), bad sidecar (missing `pixel_nm`, unknown key, `pixel_nm` 2.0, non-256 tile), out-of-range simulate params, `split='train'` → each returns its intended status + `error.code`; a good request after each still succeeds. |
| `test_filesource_roundtrip.py` | Simulate a sample, write image (float32 TIFF), CAD (PNG), scribe patches and sidecar to `tmp_path`, upload through `/upload` → `/verify`; verifier verdict + reason and CNN prob equal the simulated path; provenance is LOADED FROM FILE and `truth` is null. |
| `test_endpoints_smoke.py` | Each route with TestClient: status 200, schema keys present, `/health.cnn_loaded` true, `/results` tables non-empty and every table's numbers appear verbatim in `render_tables`/`render_day0` output (no app-side formatting), CSV paths exist. |
| `test_explain.py` | Template sentences for each Δβ sign, near-miss pass/fail branches, singleton; locator returns "no site" rather than a box when inconsistent; sentences only contain numbers present in the reason. |
| `test_sources.py` | ToolSource raises NotImplementedError; FileSource never infers pixel size; SimulatedSource badge per split. |
| existing suite | unchanged and green with and without the `[app]` extra. |

Heavy cases (D2D rule fit ~21 s/dose) use dose 50 only and share a module-scoped Engine; the rule cache file is
written under `tmp_path` in tests. I'll mark the full parity grid `@pytest.mark.slow` only if the suite exceeds
a few minutes, and say so.

## 11. Open decisions (need your call)

1. **httpx** for `TestClient` (§8): add to `[app]`, or test handlers without HTTP?
2. **Seed semantics** (§4): request seed = new wafer/scribe/refs; frozen methods stay fixed from seed 0. OK?
3. **Dose grid only** (§3): Live dose restricted to the 7 Exp-2 doses because D2D has no rule elsewhere. OK, or allow
   any dose with D2D marked "not available" off-grid?
4. **File inputs ≠ 256 px**: refuse (my proposal) or warn and centre-crop?
5. **On-disk D2D rule cache** under `results/app_cache/` (git-ignored): OK, or memory only (each server start pays
   ~21 s per dose on first use)?

## 12. Risks

- Importing `experiments.harness` couples the app to the repo layout; mitigated by the root check in `__main__`.
- The locator is a heuristic; it is labelled display-only and tested to stay silent when inconsistent.
- First-use latency per dose (~21 s) could look like a hang; the UI states what is happening.
- Uploaded real images are out of domain for every method; the UI says so on every card. No claim of real-world
  validation anywhere.
