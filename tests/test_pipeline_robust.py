"""Rung 1b candidates: all off == frozen v3; clamp, guard and alignment bound behave as pre-registered. Synthetic data."""
import json

import numpy as np
import pytest
from scipy import ndimage as ndi

from experiments.v3_harness import _pc
from nanovate.config import TestbedCfg
from nanovate.modules.calibration import calibrate
from nanovate.modules.events import EventRuleCfg
from nanovate.pipeline_robust import UNVERIFIED, ClampedCalibration, RobustCfg, calibrate_robust, verify_patch_robust
from nanovate.pipeline_v3 import verify_patch_v3
from nanovate.testbed.dataset import get_layout, make_sample
from nanovate.testbed.scribe import make_scribe_patches
from nanovate.topo_utils import Window


@pytest.fixture(scope="module")
def setup():
    dec = json.load(open("results/v3/calib/decisions.json"))
    t = TestbedCfg()
    P = make_scribe_patches(t, seed=0)
    lay = get_layout("manhattan", "val", 0, t, 0)
    S = [make_sample("manhattan", "val", 0, 0, k, t, 0, layout=lay) for k in ("nuisance", "bridge", "open", "notch")]
    return _pc(dec["pipeline_cfg"]), EventRuleCfg(**dec["event_rule_cfg"]), P, S, Window(256, 16)


def test_all_off_equals_frozen_v3(setup):
    pc, rc, P, S, win = setup
    cal = calibrate(P)
    rcal = calibrate_robust(P, RobustCfg())
    for s in S:
        a = verify_patch_v3(s.image, s.cad, cal, win, pc, rc)
        b = verify_patch_robust(s.image, s.cad, rcal, win, pc, rc, RobustCfg())
        assert a.verdict == b.verdict and a.shift == b.shift
        assert a.reason["delta_betti"] == b.reason["delta_betti"]
        assert a.reason["max_sdf_energy"] == b.reason["max_sdf_energy"] and a.reason["bottleneck_max"] == b.reason["bottleneck_max"]


def test_clamp_limits_template_and_infinite_range_is_plain(setup):
    _, _, P, S, _ = setup
    cal = calibrate(P)
    inf = ClampedCalibration(**{k: getattr(cal, k) for k in cal.__dataclass_fields__})
    assert np.array_equal(inf.predict(S[0].cad), cal.predict(S[0].cad))
    r = calibrate_robust(P, RobustCfg(clamp=(0.5, 99.5)))
    T = r.cal.predict(S[0].cad)
    assert T.min() >= r.clamp_range[0] and T.max() <= r.clamp_range[1]


def test_guard_passes_synthetic_scribe_and_fails_inconsistent_contrast(setup):
    _, _, P, _, _ = setup
    assert calibrate_robust(P, RobustCfg(guard=True)).reliable
    bad = [P[0], P[1], (P[2][0], P[2][1].max() + P[2][1].min() - P[2][1])]   # third patch with inverted contrast
    assert not calibrate_robust(bad, RobustCfg(guard=True)).reliable


def test_guard_sets_unverified_and_keeps_unguarded_verdict(setup):
    pc, rc, P, S, win = setup
    bad = [P[0], P[1], (P[2][0], P[2][1].max() + P[2][1].min() - P[2][1])]
    rcal = calibrate_robust(bad, RobustCfg(guard=True))
    r = verify_patch_robust(S[1].image, S[1].cad, rcal, win, pc, rc, RobustCfg(guard=True))
    assert r.verdict == UNVERIFIED and r.reason["verdict_unguarded"] != UNVERIFIED


@pytest.mark.parametrize("d,rejected", [(5.0, False), (20.0, True)])
def test_alignment_bound(setup, d, rejected):
    pc, rc, P, S, win = setup
    rcal = calibrate_robust(P, RobustCfg())
    img = ndi.shift(S[0].image, (d, 0.0), order=1, mode="nearest")
    r = verify_patch_robust(img, S[0].cad, rcal, win, pc, rc, RobustCfg(max_shift=12.0))
    assert r.reason["align_rejected"] is rejected
    if rejected:
        assert r.shift == (0.0, 0.0)


def test_real_rung1b_run_and_summary_smoke_on_synthetic_lattice():
    """The Rung 1b run path end to end on rendered synthetic hole arrays (no real data), both group kinds."""
    import pandas as pd
    from experiments import real_rung1 as R1
    from experiments import real_rung1b as R
    from nanovate.real.lattice import Lattice, render
    from nanovate.real.rung1 import RealImage, Reference, build_reference, tile_grid

    rng = np.random.default_rng(0)
    lat = Lattice((0.0, 80.0), (80.0, 0.0), (40.0, 40.0), 26.0)
    ims = {}
    for k in ("a.tif", "b.tif"):
        img = ndi.gaussian_filter(np.where(render(lat, (512, 768)), 140.0, 30.0), 1.5) + rng.normal(0, 3, (512, 768))
        ims[k] = RealImage(k, "G29", k, 5.0, "test", img, 130.0, dict(sha256="x"), None)
    a1, a2 = np.array(lat.a1) * 5.0, np.array(lat.a2) * 5.0
    g = R1._group("G29", ims, ["a.tif"], (a1, a2), None, [("a.tif", (0, 0)), ("a.tif", (0, 2)), ("a.tif", (1, 0))],
                  [("b.tif", rc) for rc in tile_grid((512, 768))]) | dict(refs={k: build_reference(im, a1, a2, None) for k, im in ims.items()})
    one = Lattice((0.0, 1e7), (1e7, 0.0), (256.0, 384.0), 60.0)
    fims = {}
    for k in ("c.tif", "d.tif", "e.tif", "f.tif"):
        img = ndi.gaussian_filter(np.where(render(one, (512, 768)), 140.0, 30.0), 1.5) + rng.normal(0, 3, (512, 768))
        ov = np.zeros((512, 768), bool); ov[R.OVERLAY_BOX] = True
        fims[k] = RealImage(k, "FRESH", k, 2.0, "test", img, 120.0, dict(sha256="x"), ov)
    f = dict(name="FRESH", images=fims, refs={k: Reference(one, "design", (0.0,), 1) for k in fims}, cal_ids=["c.tif", "d.tif", "e.tif"],
             eval_tiles=[("f.tif", rc) for rc in tile_grid((512, 768))])
    pc, rc = R1.frozen_cfg()
    rcfgs = {"primary": RobustCfg(guard=True, clamp=(0.5, 99.5), max_shift=12.0), "secondary_no_guard": RobustCfg(clamp=(0.5, 99.5), max_shift=12.0)}
    labels = {k: "clean" for k in list(ims) + list(fims)}
    t1, c1 = R.run([g], rcfgs, labels, pc, rc)
    t2, c2 = R.run([f], rcfgs, labels, pc, rc)
    t1["population"], t2["population"] = "orig", "fresh"
    t = pd.concat([t1, t2])
    assert len(t) == 24 and t[t.population == "fresh"].overlay.sum() == 4 and len(pd.concat([c1, c2])) == 4
    s = R.summarise(t)
    assert {"coverage (verified fraction)", "KILLER among verified clean tiles"} <= set(s.metric)


def test_real_rung1c_summary_counts_near_miss_on_clean_overlay_free_tiles():
    import pandas as pd
    from experiments import real_rung1c as C
    t = pd.DataFrame(dict(population="orig", group="G29", image_id=["a", "a", "b", "b"], tile=["0,0", "0,1", "0,0", "0,1"],
                          clean=True, overlay=[False, False, False, True], verdict=["NEAR_MISS", "NUISANCE", "KILLER_BRIDGE", "NEAR_MISS"],
                          killer=[False, False, True, False], verified=True))
    s = C.summarise(t).set_index(["population", "metric"])
    assert s.loc[("orig", "NEAR_MISS on clean tiles"), "mean"] == pytest.approx(1 / 3)
    assert s.loc[("orig", "KILLER on clean tiles"), "n_tiles"] == 3
