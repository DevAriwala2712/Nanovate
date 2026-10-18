"""FastAPI app: /api/v1 routes plus the static frontend, on localhost. Synthetic data unless loaded from file.

Every response carries request_id, commit and config_hash; analysis responses echo the seed. Errors are
``{"error": {code, message, field, request_id}}`` with a 4xx status; unexpected failures are logged with the
request id and returned as a 500 without a traceback. One bad request never stops the server.
"""
from __future__ import annotations

import base64
import io
import logging
import uuid
from importlib import metadata
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..config import TestbedCfg
from .engine import Engine, finite
from .explain import explain, locate
from .schemas import API_VERSION, HealthResponse, SimulateParams, SimulateResponse, VerifyRequest, VerifyResponse
from .sources import DOSE_GRID, InputError, InspectionInput, SimulatedSource

log = logging.getLogger("nanovate.app")
MAX_BODY = 20 * 1024 * 1024        # bytes, whole request
SYNTHETIC_NOTE = ("Synthetic data: simulated inputs come from the Nanovate testbed. The app analyses one patch at a time; "
                  "it does not scan wafers or control any tool. Nothing here is validated on real SEM images.")
RANGE_ERRORS = {"greater_than_equal", "less_than_equal", "greater_than", "less_than", "too_short", "too_long", "literal_error", "finite_number"}


def png(arr: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> str:
    """8-bit grayscale PNG data URI of ``arr`` mapped linearly from [lo, hi] (deterministic encoding)."""
    a = np.clip((np.asarray(arr, dtype=float) - lo) / max(hi - lo, 1e-12), 0.0, 1.0) * 255.0
    buf = io.BytesIO()
    Image.fromarray(np.round(a).astype(np.uint8), "L").save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def b64_array(a: np.ndarray) -> dict:
    a = np.ascontiguousarray(a, dtype=np.float32)
    return {"dtype": "float32", "shape": list(a.shape), "data": base64.b64encode(a.tobytes()).decode()}


def decode_array(d: dict) -> np.ndarray:
    return np.frombuffer(base64.b64decode(d["data"]), dtype=np.float32).reshape(d["shape"])


def sem_range(inp: InspectionInput) -> tuple[float, float]:
    """Display range of an SEM image, e-/px: the export_ui convention when the dose is known, else percentiles."""
    if inp.meta.dose is not None:
        s = TestbedCfg().sem
        return 0.0, float(inp.meta.dose * (s.a0 + s.a1) * 1.4)
    lo, hi = np.percentile(inp.image, [0.5, 99.5])
    return float(lo), float(hi)


def killer_class(verdict: str) -> str:
    return "KILLER" if verdict.startswith("KILLER") else verdict


def _err(status: int, code: str, message: str, request: Request, field: str | None = None) -> JSONResponse:
    rid = getattr(request.state, "request_id", "-")
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message, "field": field, "request_id": rid}},
                        headers={"X-Request-ID": rid})


def create_app(engine: Engine | None = None, frontend: str | Path | None = "frontend") -> FastAPI:
    app = FastAPI(title="Nanovate demonstrator", version=API_VERSION, docs_url="/api/v1/docs", openapi_url="/api/v1/openapi.json")
    state: dict = {"engine": engine}

    def eng() -> Engine:
        if state["engine"] is None:
            state["engine"] = Engine()
        return state["engine"]

    try:
        version = metadata.version("nanovate")
    except metadata.PackageNotFoundError:
        version = "unknown"

    @app.middleware("http")
    async def guard(request: Request, call_next):
        request.state.request_id = uuid.uuid4().hex[:12]
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                if int(cl) > MAX_BODY:
                    return _err(413, "too_large", f"request body {int(cl)} bytes exceeds the {MAX_BODY} byte cap", request)
            except ValueError:
                return _err(400, "bad_request", "invalid Content-Length header", request)
        try:
            response = await call_next(request)
        except Exception:  # noqa: BLE001 (logged with the request id; the server keeps serving)
            log.exception("unhandled error, request_id=%s path=%s", request.state.request_id, request.url.path)
            return _err(500, "internal", f"internal error, request id {request.state.request_id} (see the server log)", request)
        response.headers["X-Request-ID"] = request.state.request_id
        if not request.url.path.startswith("/api/"):   # frontend files: revalidate so an updated UI is never served stale
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.exception_handler(InputError)
    async def _input_error(request: Request, e: InputError):
        return _err(e.status, e.code, e.message, request, e.field)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, e: RequestValidationError):
        errs = e.errors()
        first = errs[0] if errs else {}
        loc = [str(x) for x in first.get("loc", ()) if x not in ("body", "query")]
        code = "out_of_range" if first.get("type") in RANGE_ERRORS else "bad_request"
        msg = "; ".join(f"{'.'.join(str(x) for x in er.get('loc', ()) if x != 'body') or 'body'}: {er.get('msg')}" for er in errs[:5])
        return _err(422 if code == "out_of_range" else 400, code, msg, request, ".".join(loc) or None)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, e: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed"}.get(e.status_code, "bad_request")
        return _err(e.status_code, code, str(e.detail), request)

    def envelope(request: Request) -> dict:
        e = eng()
        return {"api_version": API_VERSION, "request_id": request.state.request_id, "commit": e.commit, "config_hash": e.config_hash}

    def truth_block(inp: InspectionInput) -> dict | None:
        t = inp.truth
        if t is None:
            return None
        return {"kind": t.kind, "label": t.label, "d_beta": list(t.d_beta), "applied_shift": list(t.applied_shift),
                "resamples": t.resamples, "source": "exact topology of the simulated ground-truth wafer W vs CAD"}

    def images_block(inp: InspectionInput) -> dict:
        lo, hi = sem_range(inp)
        out = {"sem": {"png": png(inp.image, lo, hi), "lo": lo, "hi": hi, "unit": "e-/px"}, "cad": {"png": png(inp.cad)}}
        if inp.truth is not None:
            out["wafer"] = {"png": png(inp.truth.W)}
            out["expected"] = {"png": png(inp.truth.expected, lo, hi), "lo": lo, "hi": hi, "unit": "e-/px"}
            out["defect_mask"] = {"png": png(inp.truth.mask)}
        return out

    def load_sim(p: SimulateParams) -> InspectionInput:
        src = SimulatedSource(p.family, p.kind, p.dose, p.ler_sigma, p.grain,
                              None if p.shift is None else (p.shift[0], p.shift[1]), p.warp, p.seed, p.split, p.layout_id, p.die)
        src.check()
        e = eng()
        if src.input_id() in e.inputs:
            return e.get(src.input_id())
        try:
            inp = src.load()
        except RuntimeError as err:   # the testbed found no topology-consistent sample (counted resamples exhausted)
            raise InputError("simulation_failed", f"the testbed could not generate this sample: {err}", "kind") from err
        e.put(inp)
        return inp

    @app.get("/api/v1/health", response_model=HealthResponse)
    def health(request: Request):
        e = eng()
        return {**envelope(request), "version": version, "cnn_loaded": e.ctx.cnn is not None, "cnn_threshold": e.ctx.cnn_thr,
                "frozen_cfg": e.config, "dose_grid": list(DOSE_GRID), "d2d_rules_cached": e.rules_cached(),
                "calibrations_cached": sorted(e.cals), "synthetic_note": SYNTHETIC_NOTE}

    @app.post("/api/v1/simulate", response_model=SimulateResponse)
    def simulate(p: SimulateParams, request: Request, include_arrays: bool = False):
        inp = load_sim(p)
        out = {**envelope(request), "input_id": inp.input_id, "seed": inp.meta.seed, "provenance": inp.meta.provenance,
               "source_type": inp.meta.source_type, "params": inp.meta.params, "images": images_block(inp),
               "truth": truth_block(inp), "warnings": list(inp.warnings)}
        if include_arrays:
            out["arrays"] = {"image": b64_array(inp.image), "cad": b64_array(inp.cad)}
        return out

    @app.post("/api/v1/verify", response_model=VerifyResponse)
    def verify(req: VerifyRequest, request: Request):
        e = eng()
        inp = load_sim(req.simulate) if req.simulate is not None else e.get(req.input_id)
        res = e.analyse(inp)
        v, cnn, d2d, cal = res["verifier"], res["cnn"], res["d2d"], res["calibration"]
        truth = truth_block(inp)
        runtime = {"calibration_fit_ms": cal["ms"], "calibration_cached": cal["cached"],
                   "d2d_rule_cached": d2d.pop("rule_cached", None),
                   "timings_ms": {"verifier": v["timings_ms"], "verifier_total": sum(v["timings_ms"].values()), "cnn": cnn["ms"],
                                  "d2d_single": d2d.get("single", {}).get("ms"), "d2d_double": d2d.get("double", {}).get("ms")}}
        if cal.get("replaced_previous"):
            runtime["calibration_replaced_previous"] = True
        verifier = {k: v[k] for k in ("verdict", "singleton", "aligner_shift", "recovered_shift", "timings_ms")}
        verifier["reason"] = {k: val for k, val in v["reason"].items() if k != "field_meta"}
        if truth is None:
            agreement = {m: "no ground truth" for m in ("verifier", "cnn", "d2d_single", "d2d_double")}
        else:
            y = truth["label"] == "KILLER"
            agreement = {"verifier": "agrees" if killer_class(v["verdict"]) == truth["label"] else "disagrees",
                         "cnn": "agrees" if cnn["flagged"] == y else "disagrees"}
            for m in ("single", "double"):
                d = d2d.get(m, {})
                agreement[f"d2d_{m}"] = ("agrees" if d["flagged"] == y else "disagrees") if "flagged" in d else "not available"
        emap, field = res["overlays"]["sdf_energy"], res["overlays"]["field"]
        e_hi = float(max(emap.max(), (e.pc.e_star if np.isfinite(e.pc.e_star) else 0.0), 1e-9))
        overlays = {"field": {"png": png(field), "lo": 0.0, "hi": 1.0, "what": f"conductor probability P ({e.pc.field_option} field) passed to Module 4"},
                    "sdf_energy": {"png": png(emap, 0.0, e_hi), "lo": 0.0, "hi": e_hi, "offset": res["overlays"]["window"][0],
                                   "what": "Module 3 SDF tension energy on the central window (display; max feeds the near-miss rule)"},
                    "cad_outline": {"png": png(inp.cad ^ ndi.binary_erosion(inp.cad, border_value=1)), "what": "CAD conductor edges"},
                    "window": res["overlays"]["window"]}
        out = {**envelope(request), "input_id": inp.input_id, "seed": inp.meta.seed, "provenance": inp.meta.provenance,
               "source_type": inp.meta.source_type, "params": inp.meta.params, "warnings": list(inp.warnings),
               "verifier": finite(verifier), "cnn": finite({k: x for k, x in cnn.items()}), "d2d": finite(d2d), "truth": truth,
               "agreement": agreement, "calibration": finite({"layer_id": cal["layer_id"], **res["levels"]}),
               "images": images_block(inp), "overlays": overlays, "runtime": finite(runtime)}
        out["locator"] = finite(locate(field, inp.cad, v["reason"]["delta_betti"], emap, e.win))
        out["explanation"] = explain(out)
        if req.include_arrays:
            out["arrays"] = {"image": b64_array(inp.image), "cad": b64_array(inp.cad), "field": b64_array(field), "sdf_energy": b64_array(emap)}
        return out

    if frontend and Path(frontend).is_dir():
        app.mount("/", StaticFiles(directory=str(frontend), html=True), name="frontend")
    return app
