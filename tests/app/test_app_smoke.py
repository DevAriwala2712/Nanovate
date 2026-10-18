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
