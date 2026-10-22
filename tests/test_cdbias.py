"""Rung 1d: testbed CD offset and the CD-corrected SDF energy (synthetic data)."""
import json

import numpy as np
import pytest

from experiments.v3_harness import _pc
from nanovate.config import TestbedCfg
from nanovate.modules.events import EventRuleCfg
from nanovate.pipeline_robust import RobustCfg, calibrate_robust, estimate_cd_bias, verify_patch_robust
from nanovate.testbed.cdbias import grow, make_sample_cd, make_scribe_patches_cd
from nanovate.testbed.dataset import get_layout, make_sample
from nanovate.testbed.scribe import make_scribe_patches
from nanovate.topo_utils import Window


def test_grow_semantics():
    m = np.zeros((21, 21), bool); m[8:13, 8:13] = True        # 5 x 5 square
    assert np.array_equal(grow(m, 0), m)
    assert grow(m, 2).sum() > m.sum() and grow(m, 2)[6, 10] and not grow(m, 2)[5, 10]
    assert grow(m, -1).sum() == 9 and grow(m, -1)[10, 10] and not grow(m, -1)[8, 10]
    assert np.array_equal(grow(m, 0.5), m)                    # sub-pixel steps add nothing at 1 nm/px


def test_b0_reproduces_testbed_bit_for_bit():
    t = TestbedCfg()
    lay = get_layout("manhattan", "val", 0, t, 0)
    for kind in ("nuisance", "bridge"):
        a = make_sample("manhattan", "val", 0, 0, kind, t, 0, layout=lay)
        b, ok = make_sample_cd("manhattan", "val", 0, 0, kind, t, 0, 0.0, layout=lay)
        assert ok and np.array_equal(a.image, b.image) and np.array_equal(a.W, b.W) and a.topo == b.topo
    for (c1, i1), (c2, i2) in zip(make_scribe_patches(t, 0), make_scribe_patches_cd(t, 0, 0.0)):
        assert np.array_equal(c1, c2) and np.array_equal(i1, i2)


def test_offset_changes_wafer_width_and_relabels():
    t = TestbedCfg()
    lay = get_layout("manhattan", "val", 0, t, 0)
    a, _ = make_sample_cd("manhattan", "val", 0, 0, "nuisance", t, 0, 0.0, layout=lay)
    b, ok = make_sample_cd("manhattan", "val", 0, 0, "nuisance", t, 0, 2.0, layout=lay)
    assert b.W.sum() > a.W.sum() and b.params["cd_bias_nm"] == 2.0
    assert isinstance(ok, bool)


@pytest.mark.parametrize("b", [-3.0, 3.0])
def test_bias_estimate_recovers_scribe_offset(b):
    P = make_scribe_patches_cd(TestbedCfg(), 0, b)
    bh, _ = estimate_cd_bias(P, (-8.0, 8.0, 0.5))
    assert abs(bh - b) <= 1.0


def test_unbiased_scribe_gives_small_bias():
    bh, _ = estimate_cd_bias(make_scribe_patches(TestbedCfg(), 0), (-8.0, 8.0, 0.5))
    assert abs(bh) <= 1.0


def test_cd_correct_never_changes_killer_verdicts():
    dec = json.load(open("results/v3/calib/decisions.json"))
    pc, rc, win = _pc(dec["pipeline_cfg"]), EventRuleCfg(**dec["event_rule_cfg"]), Window(256, 16)
    t = TestbedCfg()
    P = make_scribe_patches_cd(t, 0, 2.0)
    base, cd = RobustCfg(clamp=(0.5, 99.5), max_shift=12.0), RobustCfg(clamp=(0.5, 99.5), max_shift=12.0, cd_correct=True)
    r0, r1 = calibrate_robust(P, base), calibrate_robust(P, cd)
    assert r1.cd_bias != 0.0 and r0.cd_bias == 0.0
    lay = get_layout("manhattan", "val", 1, t, 0)
    for kind in ("nuisance", "bridge", "open", "notch"):
        s, _ = make_sample_cd("manhattan", "val", 1, 0, kind, t, 0, 2.0, layout=lay)
        v0 = verify_patch_robust(s.image, s.cad, r0, win, pc, rc, base)
        v1 = verify_patch_robust(s.image, s.cad, r1, win, pc, rc, cd)
        assert v0.verdict.startswith("KILLER") == v1.verdict.startswith("KILLER")
        assert v0.reason["delta_betti"] == v1.reason["delta_betti"]
        assert v1.reason["max_sdf_energy_design"] == pytest.approx(v0.reason["max_sdf_energy"])


def test_real_runner_cd_correct_smoke_and_frozen_cfg_parse():
    """Real-data runner with cd_correct on a rendered hole lattice whose holes are wider than the reference."""
    from scipy import ndimage as ndi
    from experiments import real_rung1 as R1
    from experiments import real_rung1b as R
    from experiments.real_rung1d import frozen_robust
    from nanovate.real.lattice import Lattice, render
    from nanovate.real.rung1 import RealImage, build_reference, tile_grid
    rng = np.random.default_rng(0)
    real = Lattice((0.0, 80.0), (80.0, 0.0), (40.0, 40.0), 34.0)          # wafer holes 34 px
    ims = {k: RealImage(k, "G29", k, 5.0, "t", ndi.gaussian_filter(np.where(render(real, (512, 768)), 140.0, 30.0), 1.5)
                        + rng.normal(0, 3, (512, 768)), 130.0, dict(sha256="x"), None) for k in ("a.tif", "b.tif")}
    a1, a2 = np.array(real.a1) * 5.0, np.array(real.a2) * 5.0
    refs = {k: build_reference(im, a1, a2, None) for k, im in ims.items()}  # design 130 nm = 26 px: holes 8 px wider than CAD
    g = R1._group("G29", ims, ["a.tif"], (a1, a2), None, [("a.tif", (0, 0)), ("a.tif", (0, 2)), ("a.tif", (1, 0))],
                  [("b.tif", rc) for rc in tile_grid((512, 768))]) | dict(refs=refs)
    pc, rc = R1.frozen_cfg()
    rcfg = frozen_robust(dict(robust_cfg=dict(guard=True, guard_rel_spread=0.5, clamp=[0.5, 99.5], max_shift=12.0,
                                              cd_correct=True, cd_grid=[-8.0, 8.0, 0.5])))
    assert rcfg.clamp == (0.5, 99.5) and rcfg.cd_grid == (-8.0, 8.0, 0.5)
    t, c = R.run([g], {"primary": rcfg}, {"a.tif": "clean", "b.tif": "clean"}, pc, rc)
    assert len(t) == 6 and c.cd_bias_px.iloc[0] < 0                      # film (conductor) narrower than CAD
    assert t.max_sdf_energy_design.notna().all()
