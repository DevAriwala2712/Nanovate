"""Parity for the v3 pipeline (the app's default): verdicts, reasons and baselines equal direct core calls.

The direct path builds the v3 artefacts from results/v3/calib/decisions.json and calls
``nanovate.pipeline_v3.verify_patch_v3``, and separately ``experiments.v3_harness.V3Context.evaluate`` (the v3
experiments' own path) on the val split, where both use seed 0. It never goes through nanovate.app.
"""
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")

from experiments.common import ExpCfg  # noqa: E402
from experiments.harness import CNN_PATH, CNN_RULES, nuisance_cfg  # noqa: E402
from experiments.select_config import with_dose  # noqa: E402
from experiments.v3_harness import V3Context, _pc  # noqa: E402
from experiments.v3_run import render_v3  # noqa: E402
from nanovate.app.__main__ import build_parser  # noqa: E402
from nanovate.app.engine import DEFAULT_VERSION, Engine  # noqa: E402
from nanovate.baselines.cnn import load_cnn, predict_proba  # noqa: E402
from nanovate.baselines.die_to_die import d2d_flag, d2d_score  # noqa: E402
from nanovate.config import TestbedCfg  # noqa: E402
from nanovate.modules.calibration import calibrate  # noqa: E402
from nanovate.modules.events import EventRuleCfg  # noqa: E402
from nanovate.pipeline_v3 import verify_patch_v3  # noqa: E402
from nanovate.testbed.dataset import KINDS, get_layout, make_sample, window  # noqa: E402
from nanovate.testbed.scribe import make_scribe_patches  # noqa: E402

DEC = json.loads(Path("results/v3/calib/decisions.json").read_text())
PC3, RC3 = _pc(DEC["pipeline_cfg"]), EventRuleCfg(**DEC["event_rule_cfg"])
WIN = window(TestbedCfg())
CNN, _, _ = load_cnn(CNN_PATH)
THR = float(json.loads(Path(CNN_RULES).read_text())["cnn_threshold"])
REASON_KEYS = ("rule", "delta_betti", "n_bridge_events", "n_open_events", "bottleneck_max", "max_sdf_energy", "peak_z",
               "tau", "d_star", "e_star", "z_star", "singleton")


def _same(a, b):
    if isinstance(b, (tuple, list)):
        return list(a) == list(b)
    if isinstance(b, float) and not np.isfinite(b):
        return a is None
    return a == b


def direct_v3(family, kind, seed, split="live", dose=50.0, ler=1.0, grain=0.15, rules=None):
    tcfg = nuisance_cfg(with_dose(TestbedCfg(), dose), ler, grain)
    L = get_layout(family, split, 0, tcfg, seed)
    s = make_sample(family, split, 0, 0, kind, tcfg, seed, layout=L)
    refs = [make_sample(family, split, 0, 900 + j, "nuisance", tcfg, seed, layout=L).image for j in range(2)]
    cal = calibrate(make_scribe_patches(tcfg, seed=seed))
    out = {"r": verify_patch_v3(s.image, s.cad, cal, WIN, PC3, RC3), "p": float(predict_proba(CNN, s.image[None], s.cad[None], 1)[0]), "s": s}
    if rules is not None:
        out["d2d_single"] = d2d_flag(d2d_score(s.image, refs[:1], s.cad, cal, "single"), rules["single"], WIN)
        out["d2d_double"] = d2d_flag(d2d_score(s.image, refs, s.cad, cal, "double"), rules["double"], WIN)
    return out


def test_default_version_is_v3():
    assert DEFAULT_VERSION == "v3" and build_parser().parse_args([]).pipeline_version == "v3"
    assert build_parser().parse_args(["--pipeline-version", "v2"]).pipeline_version == "v2"
    with pytest.raises(ValueError):
        Engine(rule_cache=None, version="v1")


def test_frozen_config_is_the_v3_decision(engine_v3, client_v3):
    assert engine_v3.version == "v3" and engine_v3.pc == PC3 and engine_v3.rc == RC3 and engine_v3.ctx.cnn_thr == THR
    h = client_v3.get("/api/v1/health").json()
    assert h["pipeline_version"] == "v3" and h["frozen_cfg"]["pipeline_cfg_source"] == "results/v3/calib/decisions.json"
    assert h["frozen_cfg"]["verifier_call"] == "nanovate.pipeline_v3.verify_patch_v3"


@pytest.mark.parametrize("seed", [0, 7])
@pytest.mark.parametrize("kind", KINDS)
def test_verify_v3_equals_direct_core_calls(client_v3, engine_v3, seed, kind):
    api = client_v3.post("/api/v1/verify", json={"simulate": {"family": "manhattan", "kind": kind, "seed": seed}}).json()
    d = direct_v3("manhattan", kind, seed, rules=engine_v3.ctx.d2d_rules(50.0))
    v = api["verifier"]
    assert api["pipeline_version"] == v["pipeline_version"] == "v3"
    assert v["verdict"] == d["r"].verdict
    for k in REASON_KEYS:
        assert _same(v["reason"][k], d["r"].reason[k]), k
    assert v["aligner_shift"] == list(d["r"].shift)
    assert api["cnn"]["prob"] == d["p"] and api["cnn"]["flagged"] == (d["p"] >= THR)
    assert (api["d2d"]["single"]["flagged"], api["d2d"]["single"]["max_area"]) == (bool(d["d2d_single"][0]), d["d2d_single"][1])
    assert (api["d2d"]["double"]["flagged"], api["d2d"]["double"]["max_area"]) == (bool(d["d2d_double"][0]), d["d2d_double"][1])
    assert api["truth"]["label"] == d["s"].label
    assert api["explanation"]["verifier"]                           # v3 sentences, built from the event reason


def test_verify_v3_equals_v3_harness_rows(client_v3, engine_v3):
    """Val split, seed 0: the response equals V3Context.evaluate (frozen primary rule) for every kind."""
    ctx = V3Context(ExpCfg.full("v3"))
    assert ctx.d2d_rules(50.0) == engine_v3.ctx.d2d_rules(50.0)
    tcfg = with_dose(TestbedCfg(), 50.0)
    for kind in KINDS:
        row = ctx.evaluate("manhattan", "val", 0, 0, kind, tcfg)
        api = client_v3.post("/api/v1/verify", json={"simulate": {"kind": kind, "split": "val"}}).json()
        v = api["verifier"]
        assert v["verdict"] == row["verdict"], kind
        assert v["reason"]["n_bridge_events"] == row["n_bridge_events"] and v["reason"]["n_open_events"] == row["n_open_events"]
        assert v["reason"]["delta_betti"] == [row["d_b0"], row["d_b1"]]
        assert v["reason"]["bottleneck_max"] == row["bottleneck"] and v["reason"]["max_sdf_energy"] == row["energy"]
        assert api["cnn"]["prob"] == row["cnn_prob"] and api["cnn"]["flagged"] == row["cnn"]
        assert api["d2d"]["single"]["flagged"] == row["d2d_single"] and api["d2d"]["double"]["flagged"] == row["d2d_double"]
        assert api["truth"]["label"] == row["label"]


def test_evidence_shows_the_v3_tables_only(client_v3):
    r = client_v3.get("/api/v1/results").json()
    assert r["pipeline_version"] == "v3" and r["root"] == "results/v3"
    assert r["readme"]["markdown"] == render_v3(Path("results/v3")) and r["readme"]["in_sync_with_readme"] is True
    assert r["day0"]["in_sync_with_readme"] is True and "(v3)" in r["day0"]["markdown"] and "(v2)" not in r["day0"]["markdown"]
    assert "Results (v2" not in r["readme"]["markdown"] and all(p.startswith("results/v3") for p in r["readme"]["sources"] + r["day0"]["sources"])


def test_v3_report_carries_the_version_badge(client_v3):
    iid = client_v3.post("/api/v1/simulate", json={"kind": "bridge", "seed": 4}).json()["input_id"]
    j = client_v3.post("/api/v1/report", json={"input_id": iid}).json()
    assert j["pipeline_version"] == "v3" and j["verifier"]["pipeline_version"] == "v3" and j["config"]["pipeline_version"] == "v3"
    h = client_v3.post("/api/v1/report", json={"input_id": iid, "format": "html"}).text
    assert "pipeline v3" in h and "<th>pipeline version</th><td>v3</td>" in h
