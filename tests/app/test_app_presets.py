"""Demo presets (frontend/data/demo_presets.json): each still produces its documented outcome, from the val split only.

The presets are selected illustrative examples (scripts/find_demo_presets.py), not typical performance; this test
fails if a code change makes any of them stop showing what the UI says it shows.
"""
import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

PRESETS = json.loads(Path("frontend/data/demo_presets.json").read_text())
SCENARIOS = {"via_bridge_cnn_miss", "via_ler_nuisance", "manhattan_notch_near_miss", "manhattan_honest_loss"}


def test_preset_file_is_versioned_labelled_and_val_only():
    assert PRESETS["schema_version"] == 1 and PRESETS["pipeline_version"] == "v3" and PRESETS["data"] == "synthetic"
    assert "illustrative" in PRESETS["label"] and "validation split" in PRESETS["label"] and "Not typical performance" in PRESETS["label"]
    assert {p["id"] for p in PRESETS["presets"]} == SCENARIOS
    assert all(p["params"]["split"] == "val" for p in PRESETS["presets"])


@pytest.mark.parametrize("preset", PRESETS["presets"], ids=lambda p: p["id"])
def test_preset_still_produces_its_documented_outcome(client_v3, preset):
    r = client_v3.post("/api/v1/verify", json={"simulate": preset["params"]}).json()
    e = preset["expected"]
    assert r["pipeline_version"] == PRESETS["pipeline_version"]
    assert r["verifier"]["verdict"] == e["verifier_verdict"] and r["cnn"]["flagged"] == e["cnn_flagged"]
    assert r["truth"]["label"] == e["truth_label"]
    assert (r["agreement"]["verifier"] == "agrees") == e["verifier_agrees"] and (r["agreement"]["cnn"] == "agrees") == e["cnn_agrees"]
    v, fam = r["verifier"]["verdict"], preset["params"]["family"]
    {"via_bridge_cnn_miss": lambda: fam == "via" and v.startswith("KILLER") and not r["cnn"]["flagged"],
     "via_ler_nuisance": lambda: fam == "via" and preset["params"]["kind"] == "nuisance" and v == "NUISANCE",
     "manhattan_notch_near_miss": lambda: preset["params"]["kind"] == "notch" and v == "NEAR_MISS",
     "manhattan_honest_loss": lambda: fam == "manhattan" and r["agreement"]["cnn"] == "agrees" and r["agreement"]["verifier"] == "disagrees",
     }[preset["id"]]() or pytest.fail(f"{preset['id']} no longer shows its scenario")


def test_ui_labels_presets_as_illustrative_not_typical():
    html = Path("frontend/index.html").read_text()
    assert 'id="live-preset"' in html and "data/demo_presets.json" in html
    assert "selected illustrative examples, validation split" in html and "not typical performance" in html
