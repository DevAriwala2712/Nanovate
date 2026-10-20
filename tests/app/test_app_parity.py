"""Parity: the API's verdicts, reasons and baseline outputs equal direct calls to the research core.

Rule 1 of the app spec: if the app's verdict for a given (config, seed, inputs) ever differs from calling the
core directly, that is a bug. The direct path here never goes through nanovate.app.
"""
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")

from experiments.common import ExpCfg  # noqa: E402
from experiments.harness import CNN_PATH, CNN_RULES, Context, nuisance_cfg  # noqa: E402
from experiments.select_config import load_pipeline_cfg, with_dose  # noqa: E402
from nanovate.app.api import decode_array  # noqa: E402
from nanovate.baselines.cnn import load_cnn, predict_proba  # noqa: E402
from nanovate.baselines.die_to_die import d2d_flag, d2d_score  # noqa: E402
from nanovate.config import TestbedCfg  # noqa: E402
from nanovate.demo import walk_through  # noqa: E402
from nanovate.modules.calibration import calibrate  # noqa: E402
from nanovate.pipeline import verify_patch  # noqa: E402
from nanovate.testbed.dataset import KINDS, get_layout, make_sample, window  # noqa: E402
from nanovate.testbed.scribe import make_scribe_patches  # noqa: E402
from nanovate.app.engine import Engine  # noqa: E402

PC = load_pipeline_cfg(ExpCfg.full())
WIN = window(TestbedCfg())
CNN, _, _ = load_cnn(CNN_PATH)
THR = float(json.loads(Path(CNN_RULES).read_text())["cnn_threshold"])
REASON_KEYS = ("delta_betti", "beta_ref", "beta_img", "bottleneck_h0", "bottleneck_h1", "bottleneck_max", "max_sdf_energy",
               "peak_z", "tau", "d_star", "e_star", "z_star", "singleton")


def direct(family, kind, seed, split="live", dose=50.0, ler=1.0, grain=0.15, rules=None):
    """The experiments' recipe, called directly: sample, scribe calibration, verifier, CNN, D2D."""
    tcfg = nuisance_cfg(with_dose(TestbedCfg(), dose), ler, grain)
    L = get_layout(family, split, 0, tcfg, seed)
    s = make_sample(family, split, 0, 0, kind, tcfg, seed, layout=L)
    refs = [make_sample(family, split, 0, 900 + j, "nuisance", tcfg, seed, layout=L).image for j in range(2)]
    cal = calibrate(make_scribe_patches(tcfg, seed=seed))
    r = verify_patch(s.image, s.cad, cal, WIN, PC)
    p = float(predict_proba(CNN, s.image[None], s.cad[None], 1)[0])
    out = {"r": r, "p": p, "s": s}
    if rules is not None:
        out["d2d_single"] = d2d_flag(d2d_score(s.image, refs[:1], s.cad, cal, "single"), rules["single"], WIN)
        out["d2d_double"] = d2d_flag(d2d_score(s.image, refs, s.cad, cal, "double"), rules["double"], WIN)
    return out


def assert_same(api, d):
    v = api["verifier"]
    assert v["verdict"] == d["r"].verdict
    for k in REASON_KEYS:
        a, b = v["reason"][k], d["r"].reason[k]
        assert (list(a) if isinstance(a, list) else a) == (list(b) if isinstance(b, (tuple, list)) else (None if isinstance(b, float) and not np.isfinite(b) else b)), k
    assert v["singleton"] == d["r"].singleton and v["aligner_shift"] == list(d["r"].shift)
    assert api["cnn"]["prob"] == d["p"] and api["cnn"]["flagged"] == (d["p"] >= THR) and api["cnn"]["threshold"] == THR
    if "d2d_single" in d:
        assert (api["d2d"]["single"]["flagged"], api["d2d"]["single"]["max_area"]) == (bool(d["d2d_single"][0]), d["d2d_single"][1])
        assert (api["d2d"]["double"]["flagged"], api["d2d"]["double"]["max_area"]) == (bool(d["d2d_double"][0]), d["d2d_double"][1])
    t = d["s"]
    assert api["truth"]["label"] == t.label and api["truth"]["d_beta"] == [t.topo["d_beta0"], t.topo["d_beta1"]]


def test_frozen_config_is_the_v2_decision(engine):
    assert engine.pc == PC and engine.ctx.cnn_thr == THR


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize("kind", KINDS)
def test_verify_equals_direct_core_calls(client, engine, seed, kind):
    api = client.post("/api/v1/verify", json={"simulate": {"family": "manhattan", "kind": kind, "seed": seed}}).json()
    assert api["seed"] == seed and api["provenance"] == "SIMULATED (Nanovate testbed)"
    assert_same(api, direct("manhattan", kind, seed, rules=engine.ctx.d2d_rules(50.0)))


@pytest.mark.parametrize("family,kind,dose,ler,grain", [("via", "particle", 20.0, 1.5, 0.3), ("curvy", "open", 200.0, 0.5, 0.0)])
def test_verify_equals_direct_other_families_and_settings(client, engine, family, kind, dose, ler, grain):
    """Non-default family / dose / LER / grain (each new dose fits its D2D rule once, ~20 s)."""
    api = client.post("/api/v1/verify", json={"simulate": {"family": family, "kind": kind, "seed": 3, "dose": dose,
                                                            "ler_sigma": ler, "grain": grain}}).json()
    assert_same(api, direct(family, kind, 3, dose=dose, ler=ler, grain=grain, rules=engine.ctx.d2d_rules(dose)))


def test_verify_equals_experiment_harness_rows(client, engine, rule_cache):
    """Seed 0, test split: the response equals experiments.harness.Context.evaluate (the v2 experiments' own path),
    with the D2D rule refitted from scratch by a fresh Context, and an engine reloading the on-disk rule cache agrees."""
    client.post("/api/v1/verify", json={"simulate": {"kind": "nuisance", "split": "test"}})   # ensure the dose-50 rule exists
    ctx = Context(ExpCfg.full())
    fresh = ctx.d2d_rules(50.0)
    assert fresh == engine.ctx.d2d_rules(50.0)
    reloaded = Engine(rule_cache=rule_cache, version="v2")
    assert 50.0 in reloaded.rules_cached() and reloaded.ctx._rules[50.0] == fresh
    tcfg = with_dose(TestbedCfg(), 50.0)
    for kind in KINDS:
        row = ctx.evaluate("manhattan", "test", 0, 0, kind, tcfg)
        api = client.post("/api/v1/verify", json={"simulate": {"kind": kind, "split": "test"}}).json()
        assert api["provenance"] == "SIMULATED (test split sample)"
        v = api["verifier"]
        assert v["verdict"] == row["verdict"]
        assert v["reason"]["delta_betti"] == [row["d_b0"], row["d_b1"]]
        assert v["reason"]["bottleneck_max"] == row["bottleneck"] and v["reason"]["max_sdf_energy"] == row["energy"]
        assert api["cnn"]["prob"] == row["cnn_prob"] and api["cnn"]["flagged"] == row["cnn"]
        assert api["d2d"]["single"]["flagged"] == row["d2d_single"] and api["d2d"]["double"]["flagged"] == row["d2d_double"]
        assert api["truth"]["label"] == row["label"]


def test_overlays_equal_demo_walk_through(client):
    """The returned conductor field and SDF energy map are the ones the demo (and the pipeline) compute."""
    api = client.post("/api/v1/verify", json={"simulate": {"kind": "bridge", "split": "val", "shift": [1.5, -2.0]},
                                              "include_arrays": True}).json()
    w = walk_through("bridge", family="manhattan", dose=50.0, layout_id=0, die=0, seed=0, shift=(1.5, -2.0), pc=PC, split="val")
    assert api["verifier"]["verdict"] == w.result.verdict
    np.testing.assert_array_equal(decode_array(api["arrays"]["field"]), w.field.astype(np.float32))
    np.testing.assert_allclose(decode_array(api["arrays"]["sdf_energy"]), WIN.crop(w.energy_map).astype(np.float32), rtol=1e-6)
    assert float(decode_array(api["arrays"]["sdf_energy"]).max()) == pytest.approx(api["verifier"]["reason"]["max_sdf_energy"], rel=1e-6)
