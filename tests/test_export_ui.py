"""The review-UI export uses the current pipeline (calibrated levels, alignment, SDF energy, frozen config)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import export_ui  # noqa: E402


def test_export_uses_current_pipeline_and_labels_everything_synthetic(tmp_path):
    out = tmp_path / "run.js"
    export_ui.main(["--per-cell", "1", "--dose", "100", "--out", str(out)])
    text = out.read_text()
    assert text.startswith("window.NANOVATE_RUN = ")
    run = json.loads(text[len("window.NANOVATE_RUN = "):].rstrip().rstrip(";"))
    assert run["synthetic"] is True and len(run["samples"]) == 3 * 6 * 1
    m4 = run["module4"]
    assert "calibrated" in m4["levelsSource"].lower() and "oracle" not in m4["levelsSource"].lower()
    assert "not computed" not in m4["sdfEnergy"].lower()
    for s in run["samples"]:
        assert s["verdict"] in {"KILLER_BRIDGE", "KILLER_OPEN/PARTICLE", "NEAR_MISS", "NUISANCE"}
        assert s["sdfEnergy"] is not None and s["sdfEnergy"] > 0 and s["singleton"] is True
        assert set(s["img"]) == {"cad", "wafer", "sem", "field"} and s["img"]["field"].startswith("data:image/")
