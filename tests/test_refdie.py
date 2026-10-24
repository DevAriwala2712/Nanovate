"""Phase 4 reference-die mode (synthetic data)."""
import json

import numpy as np
import pytest
from scipy import ndimage as ndi

from experiments.v3_harness import _pc
from nanovate.config import TestbedCfg
from nanovate.modules.events import EventRuleCfg
from nanovate.refdie import build_reference, verify_refdie
from nanovate.testbed.dataset import get_layout, make_sample
from nanovate.testbed.labels import topology_label
from nanovate.topo_utils import Window


@pytest.fixture(scope="module")
def setup():
    dec = json.load(open("results/v3/calib/decisions.json"))
    t = TestbedCfg()
    lay = get_layout("manhattan", "val", 8, t, 0)
    refs = [make_sample("manhattan", "val", 8, 900 + j, "nuisance", t, 0, layout=lay) for j in (0, 1)]
    return _pc(dec["pipeline_cfg"]), EventRuleCfg(**dec["event_rule_cfg"]), t, lay, refs, Window(256, 16)


def test_pseudo_cad_has_cad_topology_on_a_clean_reference(setup):
    _, _, t, lay, refs, win = setup
    ref = build_reference([r.image for r in refs])
    assert ref.cad.dtype == bool and 0.2 < ref.cad.mean() < 0.8
    # compare in the CAD frame: undo die 900's acquisition shift (image(x) ~ I(x - shift))
    back = ndi.shift(ref.cad.astype(float), tuple(-v for v in refs[0].shift), order=0, mode="nearest") > 0.5
    assert topology_label(back, lay.cad, win)["topo"] == "none"


def test_alignment_of_second_reference_recovers_relative_shift(setup):
    _, _, _, _, refs, _ = setup
    ref = build_reference([r.image for r in refs])
    want = np.subtract(refs[0].shift, refs[1].shift)
    assert np.allclose(ref.shifts[1], want, atol=0.5)


def test_refdie_mode_calls_a_bridge_and_passes_a_nuisance(setup):
    pc, rc, t, lay, refs, win = setup
    imgs = [r.image for r in refs]
    br = make_sample("manhattan", "val", 8, 0, "bridge", t, 0, layout=lay)
    nu = make_sample("manhattan", "val", 8, 1, "nuisance", t, 0, layout=lay)
    rb, _, _ = verify_refdie(br.image, imgs, win, pc, rc)
    rn, _, _ = verify_refdie(nu.image, imgs, win, pc, rc)
    assert rb.verdict.startswith("KILLER")
    assert not rn.verdict.startswith("KILLER")


def test_oracle_arm_uses_given_cad(setup):
    pc, rc, t, lay, refs, win = setup
    nu = make_sample("manhattan", "val", 8, 1, "nuisance", t, 0, layout=lay)
    r, ref, _ = verify_refdie(nu.image, [refs[0].image], win, pc, rc, oracle_cad=refs[0].W)
    assert np.array_equal(ref.cad, refs[0].W) and r.verdict in ("NUISANCE", "NEAR_MISS", "KILLER_BRIDGE", "KILLER_OPEN/PARTICLE")
