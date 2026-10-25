"""Reference-cell mode on synthetic 'cells': same-type instances pass, a swapped cell is flagged."""
import json

import numpy as np
from scipy import ndimage as ndi

from experiments.v3_harness import _pc
from nanovate.modules.events import EventRuleCfg
from nanovate.pipeline_robust import RobustCfg
from nanovate.refcell import build_cell_reference, verify_cell
from nanovate.topo_utils import Window

BOX = (64, 192, 48, 208)


def _cell(pattern: str) -> np.ndarray:
    m = np.zeros((256, 256), bool)
    bars = (60, 88, 116, 144, 172)          # testbed-like: 14 px bars, 14 px gaps
    for c in bars if pattern != "missing" else (60, 88, 144, 172):
        m[70:186, c:c + 14] = True
    if pattern == "B":                      # 'B': a short bridge between two neighbouring bars
        m[120:134, 116:144] = True
    m[10:30, :] = m[226:246, :] = True      # neighbouring rows (same everywhere)
    return m


def _render(m, rng, shift):
    img = ndi.gaussian_filter(np.where(m, 140.0, 40.0), 1.5) + rng.normal(0, 4, m.shape)
    return ndi.shift(img, shift, order=1, mode="nearest")


def test_same_type_passes_swapped_type_is_killer():
    dec = json.load(open("results/v3/calib/decisions.json"))
    pc, rc = _pc(dec["pipeline_cfg"]), EventRuleCfg(**dec["event_rule_cfg"])
    rcfg = RobustCfg(guard=True, clamp=(0.5, 99.5), max_shift=12.0)
    rng = np.random.default_rng(0)
    refs = [_render(_cell("A"), rng, rng.uniform(-2, 2, 2)) for _ in range(9)]
    ref = build_cell_reference(refs, rcfg)
    win = Window(256, 16)
    ok, _ = verify_cell(_render(_cell("A"), rng, (1.0, -1.5)), ref, BOX, win, pc, rc, rcfg)
    bad, _ = verify_cell(_render(_cell("B"), rng, (0.5, 0.5)), ref, BOX, win, pc, rc, rcfg)
    assert not ok.verdict.startswith("KILLER")
    assert bad.verdict.startswith("KILLER")


def test_median_reference_survives_one_modified_reference():
    rcfg = RobustCfg(guard=True, clamp=(0.5, 99.5), max_shift=12.0)
    rng = np.random.default_rng(1)
    refs = [_render(_cell("A"), rng, (0, 0)) for _ in range(8)] + [_render(_cell("B"), rng, (0, 0))]
    ref = build_cell_reference(refs, rcfg)
    assert not ref.cad[127, 135]           # the bridge of the one 'B' reference is voted out
    assert ref.rcal.reliable


def test_missing_feature_behaviour_is_documented():
    """v3 limitation carried into Rung 3: a vanished component is not a v3 killer event (merges need pair_dist 30 px)."""
    dec = json.load(open("results/v3/calib/decisions.json"))
    pc, rc = _pc(dec["pipeline_cfg"]), EventRuleCfg(**dec["event_rule_cfg"])
    rcfg = RobustCfg(guard=True, clamp=(0.5, 99.5), max_shift=12.0)
    rng = np.random.default_rng(2)
    ref = build_cell_reference([_render(_cell("A"), rng, (0, 0)) for _ in range(9)], rcfg)
    r, _ = verify_cell(_render(_cell("missing"), rng, (0, 0)), ref, BOX, Window(256, 16), pc, rc, rcfg)
    assert r.reason["delta_betti"][0] == -1     # topology sees the missing bar ...
    assert r.verdict in ("KILLER_BRIDGE", "NEAR_MISS", "NUISANCE")   # ... recorded, whatever the v3 verdict is
