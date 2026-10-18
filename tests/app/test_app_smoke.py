"""Smoke test of each milestone-2 endpoint with FastAPI's TestClient."""
import pytest

pytest.importorskip("fastapi")


def test_health(client):
    h = client.get("/api/v1/health").json()
    assert h["cnn_loaded"] is True and h["api_version"] == "v1" and len(h["config_hash"]) == 16 and h["commit"]
    assert h["frozen_cfg"]["pipeline_cfg_source"] == "results/v2/calib/decisions.json"
    assert "does not scan wafers" in h["synthetic_note"]


def test_simulate(client):
    s = client.post("/api/v1/simulate", json={"family": "curvy", "kind": "open", "seed": 5}).json()
    assert s["provenance"] == "SIMULATED (Nanovate testbed)" and s["seed"] == 5 and s["truth"]["label"] == "KILLER"
    assert set(s["images"]) == {"sem", "cad", "wafer", "expected", "defect_mask"}
    assert all(v["png"].startswith("data:image/png;base64,") for v in s["images"].values())


def test_verify_shape(client):
    v = client.post("/api/v1/verify", json={"simulate": {"kind": "notch", "split": "val"}}).json()
    assert v["provenance"] == "SIMULATED (val split sample)"
    for k in ("verifier", "cnn", "d2d", "truth", "agreement", "calibration", "images", "overlays", "runtime", "seed", "commit", "config_hash"):
        assert k in v
    assert set(v["verifier"]["timings_ms"]) == {"align", "sdf_energy", "topology"}
    assert v["calibration"]["layer_id"].startswith("sim/dose50/")
    assert set(v["agreement"].values()) <= {"agrees", "disagrees", "not available"}


def test_frontend_is_served_with_live_mode_and_no_stale_text(engine):
    from fastapi.testclient import TestClient
    from nanovate.app.api import create_app

    c = TestClient(create_app(engine, frontend="frontend"))
    html = c.get("/").text
    assert 'data-mode="live"' in html and 'id="live-cards"' in html and "validation split, synthetic" in html
    for stale in ("are not built yet", "Module 3 not built", "Module 4 only", "untuned Module 4", "--tau 0.45"):
        assert stale not in html, stale
    assert c.get("/data/run.js").text.startswith("window.NANOVATE_RUN")
    assert c.get("/api/v1/health").json()["cnn_loaded"] is True   # API routes still win over the static mount
