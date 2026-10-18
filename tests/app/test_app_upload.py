"""FileSource and /upload: round trip, GDS CAD, and every validation error returns its intended 4xx."""
import io
import json

import numpy as np
import pytest

pytest.importorskip("fastapi")

import tifffile  # noqa: E402
from conftest import strip_timings  # noqa: E402
from PIL import Image  # noqa: E402

from nanovate.app.sources import FileSource, SimulatedSource  # noqa: E402

SIM = dict(kind="bridge", seed=3)


def tif(a, dtype=np.float32):
    b = io.BytesIO(); tifffile.imwrite(b, np.asarray(a, dtype)); return b.getvalue()


def png(m, mode="L", scale=255):
    b = io.BytesIO(); Image.fromarray((np.asarray(m) * scale).astype(np.uint8), mode).save(b, "PNG"); return b.getvalue()


@pytest.fixture(scope="module")
def sim():
    return SimulatedSource(**SIM).load()


def upload_files(inp, image=None, cad=None, scribe=True, refs=True):
    files = [("image", ("img.tif", image if image is not None else tif(inp.image))), ("cad", ("cad.png", cad if cad is not None else png(inp.cad)))]
    if scribe:
        files += [("scribe_cad", (f"sc{i}.png", png(c))) for i, (c, _) in enumerate(inp.calibration.scribe)]
        files += [("scribe_image", (f"si{i}.tif", tif(im))) for i, (_, im) in enumerate(inp.calibration.scribe)]
    if refs:
        files += [("reference", (f"ref{i}.tif", tif(r))) for i, r in enumerate(inp.references)]
    return files


SC = {"pixel_nm": 1.0, "layer_id": "upload-test", "dose": 50.0}


def post(client, files, sidecar=SC, **data):
    d = dict(data)
    if sidecar is not None:
        d["sidecar_json"] = sidecar if isinstance(sidecar, str) else json.dumps(sidecar)
    return client.post("/api/v1/upload", files=files, data=d)


def test_round_trip_verdict_unchanged(client, sim):
    """Save a simulated sample to files, load it through FileSource: same verdict, reason, CNN and D2D outputs."""
    up = post(client, upload_files(sim)).json()
    assert up["provenance"] == "LOADED FROM FILE" and up["truth"] is None and up["seed"] == 0
    f = client.post("/api/v1/verify", json={"input_id": up["input_id"]}).json()
    s = client.post("/api/v1/verify", json={"simulate": SIM}).json()
    assert f["verifier"]["verdict"] == s["verifier"]["verdict"] and f["verifier"]["reason"] == s["verifier"]["reason"]
    assert f["cnn"]["prob"] == s["cnn"]["prob"]
    assert strip_timings(f["d2d"]) == strip_timings(s["d2d"])
    assert f["truth"] is None and set(f["agreement"].values()) == {"no ground truth"}
    assert f["cnn"]["domain_note"] and any("no ground truth" in w for w in f["warnings"])
    assert f["params"]["files"]["image"]["name"] == "img.tif"


def test_round_trip_with_cached_calibration_id_and_no_references(client, sim):
    client.post("/api/v1/verify", json={"simulate": SIM})            # caches the simulated layer's calibration
    up = post(client, upload_files(sim, scribe=False, refs=False), calibration_id=sim.meta.layer_id)
    assert up.status_code == 200, up.text
    f = client.post("/api/v1/verify", json={"input_id": up.json()["input_id"]}).json()
    s = client.post("/api/v1/verify", json={"simulate": SIM}).json()
    assert f["verifier"]["reason"] == s["verifier"]["reason"]
    assert f["d2d"] == {"available": False, "why": "not available: no reference die"}
    assert any("No reference die" in w for w in f["warnings"])


def test_png_image_with_intensity_scale_and_16_bit(client, sim):
    """16-bit PNG + intensity_scale gives the same counts as the TIFF (counts here are integer-valued + read noise, so
    round them first and compare the verdict, not bit equality)."""
    q = np.clip(np.round(sim.image * 10), 0, 65535).astype(np.uint16)
    b = io.BytesIO(); Image.fromarray(q).save(b, "PNG")
    r = post(client, upload_files(sim, image=b.getvalue()), sidecar={**SC, "intensity_scale": 0.1})
    assert r.status_code == 200, r.text
    v = client.post("/api/v1/verify", json={"input_id": r.json()["input_id"]}).json()
    assert v["verifier"]["verdict"] == "KILLER_BRIDGE"


def test_gds_cad_equals_the_raster(sim):
    import gdstk

    from nanovate.config import TestbedCfg
    from nanovate.testbed.dataset import get_layout
    from nanovate.app.sources import sim_tcfg

    L = get_layout("manhattan", "live", 0, sim_tcfg(50.0, 1.0, 0.15), 3)
    lib = gdstk.Library(unit=1e-9, precision=1e-12)
    cell = lib.new_cell("TILE")
    for p in L.polygons:
        cell.add(gdstk.Polygon(p.tolist(), layer=7, datatype=0))
    b = io.BytesIO()
    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".gds"); os.close(fd)
    lib.write_gds(path); data = open(path, "rb").read(); os.unlink(path)
    pad = TestbedCfg().layout.pad
    sc = json.dumps({**SC, "gds_cell": "TILE", "gds_layer": 7, "origin_nm": [pad, pad]}).encode()
    inp = FileSource(tif(sim.image), data, sc, calibration_id="any").load()
    np.testing.assert_array_equal(inp.cad, sim.cad)


@pytest.mark.parametrize("case,status,code,field", [
    ("wrong_shape", 422, "tile_size_out_of_domain", "image"),
    ("mismatch", 422, "shape_mismatch", "cad"),
    ("rgb_png", 422, "bad_dtype", "image"),
    ("nan", 422, "non_finite", "image"),
    ("empty_cad", 422, "empty_cad", "cad"),
    ("no_calibration", 422, "calibration_missing", "calibration"),
    ("two_scribes", 422, "calibration_missing", "scribe"),
    ("unknown_cal_id", 422, "unknown_calibration_id", "calibration_id"),
    ("oversized_file", 413, "too_large", "image"),
    ("huge_dims", 413, "too_large", "image"),
    ("jpeg", 415, "unsupported_format", "image"),
    ("garbage_png", 415, "unsupported_format", "image"),
    ("no_sidecar", 422, "bad_sidecar", "sidecar"),
    ("sidecar_not_json", 422, "bad_sidecar", None),
    ("sidecar_missing_pixel", 422, "bad_sidecar", "pixel_nm"),
    ("sidecar_unknown_key", 422, "bad_sidecar", "colour"),
    ("sidecar_pixel_2nm", 422, "pixel_size_out_of_domain", "pixel_nm"),
    ("png_without_scale", 422, "bad_sidecar", "intensity_scale"),
    ("reserved_layer_id", 422, "bad_sidecar", "layer_id"),
    ("ref_mismatch", 422, "shape_mismatch", "reference[0]"),
    ("missing_image", 400, "bad_request", "image"),
])
def test_upload_validation(client, sim, case, status, code, field):
    files, sc, data = upload_files(sim), dict(SC), {}
    big = np.zeros((300, 300), np.float32)
    if case == "wrong_shape":
        files[0] = ("image", ("i.tif", tif(big))); files[1] = ("cad", ("c.png", png(big > -1)))
    elif case == "mismatch":
        files[1] = ("cad", ("c.png", png(np.zeros((128, 128)))))
    elif case == "rgb_png":
        b = io.BytesIO(); Image.new("RGB", (256, 256)).save(b, "PNG"); files[0] = ("image", ("i.png", b.getvalue())); sc["intensity_scale"] = 1.0
    elif case == "nan":
        im = sim.image.copy(); im[5, 5] = np.nan; files[0] = ("image", ("i.tif", tif(im)))
    elif case == "empty_cad":
        files[1] = ("cad", ("c.png", png(np.zeros((256, 256)))))
    elif case == "no_calibration":
        files = upload_files(sim, scribe=False)
    elif case == "two_scribes":
        files = [f for f in files if f[1][0] not in ("sc2.png", "si2.tif")]
    elif case == "unknown_cal_id":
        files = upload_files(sim, scribe=False); data["calibration_id"] = "no-such-layer"
    elif case == "oversized_file":
        files[0] = ("image", ("i.tif", b"II*\x00" + b"\x00" * (8 * 1024 * 1024 + 10)))
    elif case == "huge_dims":
        files[0] = ("image", ("i.tif", tif(np.zeros((2100, 16), np.uint8), np.uint8)))
    elif case == "jpeg":
        b = io.BytesIO(); Image.new("L", (256, 256)).save(b, "JPEG"); files[0] = ("image", ("i.jpg", b.getvalue()))
    elif case == "garbage_png":
        files[0] = ("image", ("i.png", b"\x89PNG\r\n\x1a\n" + b"junk" * 50)); sc["intensity_scale"] = 1.0
    elif case == "no_sidecar":
        sc = None
    elif case == "sidecar_not_json":
        sc = "{pixel_nm: 1"
    elif case == "sidecar_missing_pixel":
        sc.pop("pixel_nm")
    elif case == "sidecar_unknown_key":
        sc["colour"] = "blue"
    elif case == "sidecar_pixel_2nm":
        sc["pixel_nm"] = 2.0
    elif case == "png_without_scale":
        files[0] = ("image", ("i.png", png(np.clip(sim.image / 100, 0, 1))))
    elif case == "reserved_layer_id":
        sc["layer_id"] = "sim-fake"
    elif case == "ref_mismatch":
        files = upload_files(sim, refs=False) + [("reference", ("r.tif", tif(np.zeros((128, 128)))))]
    elif case == "missing_image":
        files = files[1:]
    r = post(client, files, sidecar=sc, **data)
    assert r.status_code == status, r.text
    e = r.json()["error"]
    assert e["code"] == code and e["message"], e
    if field:
        assert field in (e["field"] or ""), e
    assert client.get("/api/v1/health").status_code == 200


def test_pixel_size_is_never_inferred(sim):
    """No pixel_nm -> refused, whatever the image size."""
    from nanovate.app.sources import InputError
    with pytest.raises(InputError) as e:
        FileSource(tif(sim.image), png(sim.cad), json.dumps({"layer_id": "x"}).encode(), calibration_id="c").load()
    assert e.value.code == "bad_sidecar" and e.value.field == "pixel_nm"
