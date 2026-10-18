"""Bad requests get a structured 4xx (or a logged 500) and never take the server down."""
import numpy as np
import pytest

pytest.importorskip("fastapi")

from nanovate.app.sources import CalibrationRef, InputError, InputMeta, InspectionInput, ToolSource, validate_arrays  # noqa: E402

GOOD = {"simulate": {"kind": "nuisance", "seed": 2}}


def _check(client, r, status, code, field=None):
    assert r.status_code == status, r.text
    e = r.json()["error"]
    assert e["code"] == code and e["message"] and e["request_id"] == r.headers["X-Request-ID"]
    if field:
        assert field in (e["field"] or "")
    assert client.post("/api/v1/verify", json=GOOD).status_code == 200   # the server still serves


@pytest.mark.parametrize("params,code,field", [
    ({"ler_sigma": 3.0}, "out_of_range", "ler_sigma"),
    ({"grain": -0.1}, "out_of_range", "grain"),
    ({"shift": [9.0, 0.0]}, "out_of_range", "shift"),
    ({"shift": [1.0]}, "out_of_range", "shift"),
    ({"dose": 42.0}, "out_of_range", "dose"),
    ({"family": "spiral"}, "out_of_range", "family"),
    ({"kind": "scratch"}, "out_of_range", "kind"),
    ({"split": "train"}, "split_not_allowed", "split"),
    ({"die": 900}, "out_of_range", "die"),
    ({"seed": -1}, "out_of_range", "seed"),
])
def test_simulate_params_out_of_range(client, params, code, field):
    _check(client, client.post("/api/v1/simulate", json=params), 422, code, field)


def test_bad_bodies(client):
    _check(client, client.post("/api/v1/verify", json={}), 400, "bad_request")
    _check(client, client.post("/api/v1/verify", json={"input_id": "x", "simulate": {}}), 400, "bad_request")
    _check(client, client.post("/api/v1/simulate", json={"bogus": 1}), 400, "bad_request")
    _check(client, client.post("/api/v1/simulate", content=b"{not json", headers={"content-type": "application/json"}), 400, "bad_request")
    _check(client, client.post("/api/v1/verify", json={"input_id": "nope"}), 404, "unknown_input_id", "input_id")
    _check(client, client.get("/api/v1/nothing-here"), 404, "not_found")


def test_oversized_request(client):
    _check(client, client.post("/api/v1/verify", content=b"x" * (21 * 1024 * 1024), headers={"content-type": "application/json"}),
           413, "too_large")


def test_internal_error_is_logged_with_request_id(client, engine, monkeypatch, caplog):
    def boom(_):
        raise RuntimeError("synthetic failure for the test")
    monkeypatch.setattr(engine, "analyse", boom)
    r = client.post("/api/v1/verify", json=GOOD)
    assert r.status_code == 500 and r.json()["error"]["code"] == "internal" and "Traceback" not in r.text
    assert any(r.json()["error"]["request_id"] in rec.getMessage() for rec in caplog.records)
    monkeypatch.undo()
    assert client.post("/api/v1/verify", json=GOOD).status_code == 200


IMG, CAD = np.full((256, 256), 20.0, np.float32), np.zeros((256, 256), bool)
CAD[100:120, :] = True


@pytest.mark.parametrize("image,cad,code", [
    (IMG[:200], CAD[:200], "tile_size_out_of_domain"),
    (IMG, CAD[:200], "shape_mismatch"),
    (np.stack([IMG] * 3, -1), CAD, "bad_dtype"),
    (IMG.astype(np.complex64), CAD, "bad_dtype"),
    (np.where(CAD, np.nan, IMG), CAD, "non_finite"),
    (IMG, np.zeros_like(CAD), "empty_cad"),
    (IMG, np.ones_like(CAD), "full_cad"),
    (IMG, CAD.astype(np.float32), "bad_dtype"),
])
def test_validate_arrays(image, cad, code):
    with pytest.raises(InputError) as e:
        validate_arrays(image, cad)
    assert e.value.code == code and e.value.status == 422


def test_missing_calibration_is_refused(engine):
    inp = InspectionInput(IMG, CAD, InputMeta(1.0, None, "layerX", "file", "LOADED FROM FILE", {}), CalibrationRef())
    with pytest.raises(InputError) as e:
        engine.calibration(inp)
    assert e.value.code == "calibration_missing"
    inp = InspectionInput(IMG, CAD, InputMeta(1.0, None, "layerX", "file", "LOADED FROM FILE", {}), CalibrationRef(calibration_id="nope"))
    with pytest.raises(InputError) as e:
        engine.calibration(inp)
    assert e.value.code == "unknown_calibration_id"


def test_tool_source_is_a_stub():
    with pytest.raises(NotImplementedError):
        ToolSource().load()


@pytest.mark.parametrize("body,status,code", [
    (b'{"simulate":{"shift":[NaN,0]}}', 422, "out_of_range"),        # used to crash the simulator (500)
    (b'{"simulate":{"shift":[Infinity,0]}}', 422, "out_of_range"),
    (b'{"simulate":{"ler_sigma":NaN}}', 422, "out_of_range"),
    (b'{"simulate":{"seed":true}}', 400, "bad_request"),              # used to be accepted as seed 1
    (b'{"simulate":{"dose":"50"}}', 400, "bad_request"),              # no string -> number coercion
    (b"[" * 200000, 400, "bad_request"),                               # deep nesting: no RecursionError / 500
    (b'{"simulate":' * 5000 + b"{}" + b"}" * 5000, 400, "bad_request"),
])
def test_hostile_json_bodies(client, body, status, code):
    _check(client, client.post("/api/v1/verify", content=body, headers={"content-type": "application/json"}), status, code)


@pytest.mark.parametrize("kw", [dict(ler_sigma=float("nan")), dict(grain=float("inf")), dict(warp=-0.1), dict(shift=(float("nan"), 0.0)),
                                dict(seed=-1), dict(die=900)])
def test_simulated_source_rejects_non_finite_and_out_of_range(kw):
    from nanovate.app.sources import SimulatedSource
    with pytest.raises(InputError) as e:
        SimulatedSource(**kw).check()
    assert e.value.code == "out_of_range"


def test_integer_numbers_are_still_accepted(client):
    r = client.post("/api/v1/simulate", json={"kind": "nuisance", "dose": 50, "shift": [1, -2], "warp": 0})
    assert r.status_code == 200 and r.json()["truth"]["applied_shift"] == [1.0, -2.0]


def test_frontend_files_are_revalidated(engine):
    from fastapi.testclient import TestClient
    from nanovate.app.api import create_app
    c = TestClient(create_app(engine, frontend="frontend"))
    assert c.get("/").headers["cache-control"] == "no-cache"
    assert c.get("/selftest.html").status_code == 200
    assert "cache-control" not in c.get("/api/v1/health").headers
