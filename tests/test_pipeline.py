"""End-to-end pipeline: one test per defect class, invariance (LER-only, shift), reasons, stack path.

Dose 200, Manhattan val. The defect-class samples are fixed known-good seeds (regression tests of
behaviour); recall over the whole val/test population is *measured* by the experiments, not asserted here.
"""
import dataclasses
import math

import numpy as np
import pytest

from nanovate.config import TestbedCfg
from nanovate.modules.calibration import calibrate
from nanovate.modules.topology import KILLER_BRIDGE, KILLER_OPEN_PARTICLE, NEAR_MISS, NUISANCE
from nanovate.pipeline import PipelineCfg, verify_patch, verify_stack
from nanovate.testbed.dataset import get_layout, make_sample, window
from nanovate.testbed.scribe import make_scribe_patches

SHIFTS = [(0.0, 0.0), (6.4, -4.1), (-7.3, 5.2)]


@pytest.fixture(scope="module")
def world():
    base = TestbedCfg()
    cfg = dataclasses.replace(base, sem=dataclasses.replace(base.sem, dose=200.0))
    return cfg, calibrate(make_scribe_patches(cfg, seed=0)), window(cfg)


def _verdicts(world, kind, lay, shifts=SHIFTS, pc=PipelineCfg()):
    cfg, cal, win = world
    L = get_layout("manhattan", "val", lay, cfg, 0)
    out = []
    for sh in shifts:
        s = make_sample("manhattan", "val", lay, 0, kind, cfg, 0, layout=L, shift=sh)
        out.append(verify_patch(s.image, s.cad, cal, win, pc))
    return out


@pytest.mark.parametrize("kind,lay,expected", [
    ("bridge", 0, KILLER_BRIDGE), ("bridge", 3, KILLER_BRIDGE),
    ("open", 1, KILLER_OPEN_PARTICLE), ("open", 4, KILLER_OPEN_PARTICLE),
    ("particle", 1, KILLER_OPEN_PARTICLE),
])
def test_defect_class_verdict_is_invariant_to_shift(world, kind, lay, expected):
    verdicts = [r.verdict for r in _verdicts(world, kind, lay)]
    assert verdicts == [expected] * 3, verdicts


@pytest.mark.parametrize("lay", [0, 2, 4])
def test_ler_only_sample_is_nuisance_at_any_shift(world, lay):
    assert [r.verdict for r in _verdicts(world, "nuisance", lay)] == [NUISANCE] * 3


def test_result_carries_reason_flags_and_timings(world):
    r = _verdicts(world, "bridge", 0, shifts=[(2.0, 1.0)])[0]
    for key in ("delta_betti", "beta_ref", "beta_img", "bottleneck_h0", "bottleneck_h1", "max_sdf_energy", "tau"):
        assert key in r.reason
    assert r.singleton is True and r.reason["singleton"] is True
    assert r.reason["max_sdf_energy"] is not None and r.reason["max_sdf_energy"] > 0
    assert set(r.timings_ms) == {"align", "sdf_energy", "topology"} and all(v > 0 for v in r.timings_ms.values())
    assert max(abs(r.shift[0] + 2.0), abs(r.shift[1] + 1.0)) < 1.0   # recovered shift ~ -(2, 1)


def test_near_miss_branch_needs_both_thresholds(world):
    cfg, cal, win = world
    base = [r for r in _verdicts(world, "notch", 0, shifts=[(0.0, 0.0)])][0]
    if base.verdict != NUISANCE:
        pytest.skip("notch at layout 0 reads as a killer here")
    on = _verdicts(world, "notch", 0, shifts=[(0.0, 0.0)], pc=PipelineCfg(d_star=0.0, e_star=0.0))[0]
    assert on.verdict == NEAR_MISS and on.reason["near_miss_enabled"] is True
    off = _verdicts(world, "notch", 0, shifts=[(0.0, 0.0)], pc=PipelineCfg(d_star=0.0, e_star=1e9))[0]
    assert off.verdict == NUISANCE


def test_alignment_off_is_flagged_in_cfg_and_changes_nothing_at_zero_shift(world):
    on = _verdicts(world, "bridge", 0, shifts=[(0.0, 0.0)])[0]
    off = _verdicts(world, "bridge", 0, shifts=[(0.0, 0.0)], pc=PipelineCfg(align=False))[0]
    assert off.shift == (0.0, 0.0) and off.verdict == on.verdict and "align" in off.timings_ms


def test_stack_path_is_flagged_not_singleton_and_deterministic(world):
    cfg, cal, win = world
    L = get_layout("manhattan", "val", 0, cfg, 0)
    kinds = ["nuisance"] * 11 + ["bridge"]
    S = [make_sample("manhattan", "val", 0, k, kd, cfg, 0, layout=L) for k, kd in enumerate(kinds)]
    pc = PipelineCfg(use_stack=True)
    a = verify_stack([s.image for s in S], L.cad, cal, win, pc, seed=1)
    b = verify_stack([s.image for s in S], L.cad, cal, win, pc, seed=1)
    assert len(a) == 12 and all(r.singleton is False and r.reason["singleton"] is False for r in a)
    assert [r.verdict for r in a] == [r.verdict for r in b]
    assert "stack" in a[0].timings_ms
    solo = verify_stack([S[0].image], L.cad, cal, win, pc)
    assert len(solo) == 1 and solo[0].singleton is True


def test_pipeline_cfg_roundtrip_and_defaults():
    pc = PipelineCfg()
    d = pc.to_dict()
    assert PipelineCfg(**d) == pc and math.isinf(d["d_star"]) and d["align"] is True
