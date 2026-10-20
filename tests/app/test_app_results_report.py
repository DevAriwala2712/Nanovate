"""/results reproduces the README tables from the CSVs; /report is complete, deterministic and escaped."""
import json
import re
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("fastapi")

from conftest import strip_timings  # noqa: E402

from experiments import report  # noqa: E402,F401
from nanovate.app.evidence import renderers  # noqa: E402


def test_results_tables_come_from_the_report_generator(client):
    r = client.get("/api/v1/results").json()
    assert r["synthetic"] is True and r["root"] == "results/v2" and r["pipeline_version"] == "v2"
    tables_fn, _, day0_fn, _ = renderers("v2")
    for key, fn in (("readme", tables_fn), ("day0", day0_fn)):
        block = r[key]
        assert block["markdown"] == fn(Path(r["root"]))                  # nothing reformatted by the app
        assert block["in_sync_with_readme"] is True                      # README shows exactly these tables
        assert block["sources"] and all(Path(p).exists() for p in block["sources"])
        tables = [s for s in block["sections"] if s["headers"]]
        assert tables and all(len(row) == len(s["headers"]) for s in tables for row in s["rows"])
        for s in tables:                                                 # every cell appears verbatim in the generated markdown
            for row in s["rows"]:
                for cell in row:
                    assert cell in block["markdown"], cell


def test_results_show_the_losses(client):
    L = client.get("/api/v1/results").json()["losses"]
    lat = pd.read_csv("results/v2/exp6/latency.csv").set_index("what")["median"]
    assert L["cnn_speed_ratio"]["value"] == pytest.approx(lat["verifier: total per patch (singleton path)"] / lat["cnn batch 1 (default threads)"])
    assert L["manhattan_f1"]["cnn"]["f1"] > L["manhattan_f1"]["verifier"]["f1"]        # the verifier loses on Manhattan
    curvy = [row for row in L["breakeven"]["rows"] if row["family"] == "curvy"]
    assert curvy and all(row["breakeven_k"] == "0" for row in curvy)                  # no day-0 advantage on curvy
    md = client.get("/api/v1/results").json()["day0"]["markdown"]
    assert "Where the verifier loses" in md


def test_report_json_is_complete_and_deterministic(client):
    iid = client.post("/api/v1/simulate", json={"kind": "open", "seed": 9}).json()["input_id"]
    a = client.post("/api/v1/report", json={"input_id": iid}).json()
    b = client.post("/api/v1/report", json={"input_id": iid, "format": "json"}).json()
    for k in ("label", "disclaimer", "config", "config_hash", "commit", "seed", "provenance", "verifier", "cnn", "d2d", "truth",
              "explanation", "images", "runtime"):
        assert k in a, k
    assert a["seed"] == 9 and a["label"].startswith("SYNTHETIC DATA") and "does not scan wafers" in a["disclaimer"]
    assert strip_timings(a) == strip_timings(b)
    r = client.post("/api/v1/report", json={"input_id": iid})
    assert r.headers["content-disposition"] == f'attachment; filename="nanovate_report_{iid}.json"'


def test_report_html_is_self_contained_and_escaped(client, engine):
    iid = client.post("/api/v1/simulate", json={"kind": "particle", "seed": 2}).json()["input_id"]
    inp = engine.get(iid)
    import dataclasses
    evil = dataclasses.replace(inp, warnings=("<script>alert(1)</script>",))
    engine.inputs[iid] = evil
    h = client.post("/api/v1/report", json={"input_id": iid, "format": "html"})
    assert h.status_code == 200 and h.headers["content-type"].startswith("text/html")
    t = h.text
    assert "<script>alert(1)</script>" not in t and "&lt;script&gt;" in t
    assert "SIMULATED (Nanovate testbed)" in t and "<th>seed</th><td>2</td>" in t and "<th>commit</th>" in t
    assert not re.search(r'(src|href)="https?://', t)                       # no external resources
    assert "var(--bg-deep)" in t and "--bg-deep:" in t                       # the UI's own CSS variables


def test_report_errors(client):
    r = client.post("/api/v1/report", json={"input_id": "nope"})
    assert r.status_code == 404 and r.json()["error"]["code"] == "unknown_input_id"
    r = client.post("/api/v1/report", json={"input_id": "x", "format": "pdf"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "out_of_range"
