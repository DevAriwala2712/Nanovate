"""The sample bundles from scripts/make_sample_uploads.py upload as documented: valid ones give the direct-core verdict,
broken ones are refused with the error code their README names."""
import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from scripts.make_sample_uploads import main as make_samples  # noqa: E402


@pytest.fixture(scope="module")
def bundles(tmp_path_factory):
    root = tmp_path_factory.mktemp("samples")
    make_samples(["--out", str(root)])
    return root, json.loads((root / "index.json").read_text())


def files_of(d: Path):
    pick = lambda pat: sorted(d.glob(pat))  # noqa: E731
    out = [("image", (p.name, p.read_bytes())) for p in pick("image.*")] + [("cad", (p.name, p.read_bytes())) for p in pick("cad.*")]
    out += [("sidecar", ("sidecar.json", (d / "sidecar.json").read_bytes()))]
    out += [("scribe_cad", (p.name, p.read_bytes())) for p in pick("scribe_cad_*.png")]
    out += [("scribe_image", (p.name, p.read_bytes())) for p in pick("scribe_image_*.tif")]
    out += [("reference", (p.name, p.read_bytes())) for p in pick("reference_*.tif")]
    return out


def test_every_bundle_behaves_as_documented(client, bundles):
    root, index = bundles
    assert len(index) == 13 and (root / "INDEX.md").exists()
    for row in index:
        d = root / row["bundle"]
        r = client.post("/api/v1/upload", files=files_of(d))
        if not row["valid"]:
            assert r.status_code in (413, 415, 422), (row["bundle"], r.text)
            assert r.json()["error"]["code"] == row["expected_error"], (row["bundle"], r.json())
            continue
        assert r.status_code == 200, (row["bundle"], r.text)
        v = client.post("/api/v1/verify", json={"input_id": r.json()["input_id"]}).json()
        assert v["provenance"] == "LOADED FROM FILE" and v["truth"] is None
        truth = json.loads((d / "truth.json").read_text())
        if not row["bundle"].startswith("03_"):                  # 03 is a rounded 16-bit PNG (documented)
            assert v["verifier"]["verdict"] == truth["verifier_verdict_direct_core_call"], row["bundle"]
        assert v["d2d"]["available"] == row["d2d"].startswith("available"), row["bundle"]


def test_samples_are_deterministic(bundles, tmp_path):
    root, _ = bundles
    make_samples(["--out", str(tmp_path / "again")])
    for p in root.rglob("*"):
        if p.is_file() and p.suffix != ".gds":                   # GDS stores a write timestamp
            assert p.read_bytes() == (tmp_path / "again" / p.relative_to(root)).read_bytes(), p
