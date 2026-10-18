"""v3 located-event rule (modules/events.py, pipeline_v3.py). Synthetic data."""
from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import ndimage as ndi

from nanovate.config import TestbedCfg
from nanovate.modules.calibration import calibrate
from nanovate.modules.events import EventRuleCfg, decide, evidence, located_pairs, v2_delta_beta, v3_events
from nanovate.modules.topology import KILLER_BRIDGE, NUISANCE, betti_from_intervals, compute_persistence_intervals
from nanovate.pipeline import PipelineCfg, verify_patch
from nanovate.pipeline_v3 import patch_evidence, verify_patch_v3
from nanovate.testbed.dataset import KINDS, make_sample, window
from nanovate.testbed.scribe import make_scribe_patches

PC = PipelineCfg(**json.load(open("results/v2/calib/decisions.json"))["pipeline_cfg"])
CFG = TestbedCfg()
RC = EventRuleCfg(m_merge=0.5)


@pytest.fixture(scope="module")
def cal():
    return calibrate(make_scribe_patches(CFG, seed=0))


def _lines(H=64, W=64, ys=((10, 24), (38, 52))):
    cad = np.zeros((H, W), bool)
    for a, b in ys:
        cad[a:b, :] = True
    return cad


def _smooth(cad):
    return np.clip(ndi.gaussian_filter(cad.astype(float), 1.5), 0, 1)


def test_located_pairs_put_death_at_the_neck():
    f = np.zeros((7, 9)); f[1:6, 1:3] = 1; f[1:6, 6:8] = 1; f[3, 3:6] = 0.6
    reg, ess = located_pairs(f, 0)
    assert len(reg) == 1 and len(ess) == 1
    r, c = np.unravel_index(reg[0, 1], f.shape)
    assert r == 3 and 3 <= c <= 5 and f[r, c] == 0.6


def test_clean_gap_matches_its_expectation():
    cad = _lines()
    fx = _smooth(cad)
    ev = evidence(fx, cad, np.ones_like(cad), fx)
    assert len(ev.pairs) == 1
    p = ev.pairs[0]
    assert p["s"] == pytest.approx(p["e"]) and not v3_events(ev, RC, 0.35)["bridge"]


def test_designed_narrow_gap_does_not_fire_but_a_bridge_does():
    cad = _lines(ys=((10, 24), (27, 41)))   # 3 px designed gap: high expected merge level
    fx = _smooth(cad)
    ev = evidence(fx, cad, np.ones_like(cad), fx)
    assert ev.pairs[0]["e"] > 0.2 and not v3_events(ev, RC, 0.35)["bridge"]
    f = fx.copy(); f[23:29, 30:36] = 0.95   # bright bridge across the narrow gap
    ev = evidence(f, cad, np.ones_like(cad), fx)
    assert v3_events(ev, EventRuleCfg(m_merge=0.3), 0.35)["bridge"]


def test_dim_bridge_with_an_unrelated_split_is_not_cancelled():
    cad = _lines()
    fx = _smooth(cad)
    f = fx.copy()
    f[22:40, 20:26] = 0.7     # bridge between the lines (merge level 0.7)
    f[36:54, 50:54] = 0.0     # unrelated full cut across the second line
    reg = np.ones_like(cad)
    ev = evidence(f, cad, reg, fx)
    # v2 reading: one merge and one split sum to zero
    d0, d1 = compute_persistence_intervals(np.where(reg, f, cad.astype(float)))
    b_img = betti_from_intervals(d0, compute_persistence_intervals(np.where(reg, f, 0.0))[1], 0.35)
    assert v2_delta_beta(ev, 0.35) == (b_img[0] - 2, b_img[1] - 0) == (0, 0)
    verdict, events = decide(ev, RC, 0.35, 0.0, 0.0, 0.0, np.inf, np.inf, -np.inf)
    assert verdict == KILLER_BRIDGE and events["bridge"] and events["open"]
    v2_verdict, _ = decide(ev, EventRuleCfg(rule="v2_global_tau"), 0.35, 0.0, 0.0, 0.0, np.inf, np.inf, -np.inf)
    assert v2_verdict == NUISANCE


def test_grain_normalised_merge_separates_flat_dim_bridge_from_dipped_filament():
    cad = _lines()
    fx = _smooth(cad)
    bridge = fx.copy(); bridge[22:40, 20:26] = 0.55              # uniformly dim (dark grain)
    fil = fx.copy(); fil[22:36, 20:26] = 0.95; fil[36:38, 20:26] = 0.55   # bright body, dip at the gap
    rc = EventRuleCfg(m_merge=0.4, normalise=True, m_norm=0.8)
    assert v3_events(evidence(bridge, cad, np.ones_like(cad), fx), rc, 0.35)["bridge"]
    assert not v3_events(evidence(fil, cad, np.ones_like(cad), fx), rc, 0.35)["bridge"]


@pytest.mark.parametrize("kind", KINDS)
def test_evidence_reproduces_v2_delta_beta_and_bottleneck(kind, cal):
    s = make_sample("manhattan", "val", 1, 0, kind, CFG, seed=0)
    ref = verify_patch(s.image, s.cad, cal, window(CFG), PC)
    ev, sc, _ = patch_evidence(s.image, s.cad, cal, window(CFG), PC)
    assert sc["delta_betti_v2"] == tuple(ref.reason["delta_betti"])
    assert sc["bottleneck"] == pytest.approx(ref.reason["bottleneck_max"])
    assert sc["energy"] == pytest.approx(ref.reason["max_sdf_energy"])


def test_v2_rule_selected_by_config_is_verify_patch(cal):
    s = make_sample("manhattan", "val", 0, 1, "bridge", CFG, seed=0)
    a = verify_patch(s.image, s.cad, cal, window(CFG), PC)
    b = verify_patch_v3(s.image, s.cad, cal, window(CFG), PC, EventRuleCfg(rule="v2_global_tau"))
    assert a.verdict == b.verdict and a.reason["delta_betti"] == b.reason["delta_betti"]


def test_ler_only_sample_gives_no_events(cal):
    s = make_sample("manhattan", "val", 2, 0, "nuisance", CFG, seed=0)
    r = verify_patch_v3(s.expected, s.cad, cal, window(CFG), PC, RC)
    assert r.reason["n_bridge_events"] == 0 and r.reason["n_open_events"] == 0


def test_shifted_bridge_keeps_its_verdict(cal):
    a = make_sample("manhattan", "val", 0, 0, "bridge", CFG, seed=0, shift=(0.0, 0.0))
    b = make_sample("manhattan", "val", 0, 0, "bridge", CFG, seed=0, shift=(2.5, -1.5))
    va = verify_patch_v3(a.expected, a.cad, cal, window(CFG), PC, RC).verdict
    vb = verify_patch_v3(b.expected, b.cad, cal, window(CFG), PC, RC).verdict
    assert va == vb == KILLER_BRIDGE


def test_report_v2_block_still_byte_identical_and_v3_exp3_wording(tmp_path):
    from pathlib import Path

    import pandas as pd

    from experiments import report
    readme = Path("README.md").read_text()
    blk = readme.split(report.START, 1)[1].split(report.END, 1)[0].strip("\n")
    assert report.render_tables(Path("results/v2")) == blk
    d = tmp_path / "v3" / "exp3"
    d.mkdir(parents=True)
    pd.read_csv("results/v2/exp3/f1_drop.csv").to_csv(d / "f1_drop.csv", index=False)
    out = report.render_tables(tmp_path / "v3")
    assert "via: +7.3 pts (improved)" in out and "curvy: +3.7 pts (improved)" in out
