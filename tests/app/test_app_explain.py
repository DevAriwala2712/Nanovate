"""Template explanations and the display-only locator."""
import re

import numpy as np
import pytest

pytest.importorskip("fastapi")

from nanovate.app.api import decode_array  # noqa: E402
from nanovate.app.explain import explain_cnn, explain_d2d, explain_truth, explain_verifier, locate  # noqa: E402
from nanovate.topo_utils import Window  # noqa: E402

BASE = {"tau": 0.35, "near_miss_enabled": True, "d_star": 0.0775434990066642, "e_star": 2.6189942373880593, "z_star": 5.578071347336651, "singleton": True,
        "bottleneck_max": 0.5, "max_sdf_energy": 4.61, "peak_z": 26.1}


def R(d, ref, img, **kw):
    return {**BASE, "delta_betti": list(d), "beta_ref": list(ref), "beta_img": list(img), **kw}


def test_bridge_sentence():
    t = explain_verifier("KILLER_BRIDGE", R((-1, 0), (10, 0), (9, 0)))
    assert t[0] == "β₀ dropped by 1 (CAD 10 conductors, image 9): two separate CAD conductors are connected in the image → bridge."
    assert any("Δβ rule alone" in s and "τ = 0.35" in s for s in t) and any("Singleton path" in s for s in t)


@pytest.mark.parametrize("d,needle", [((1, 0), "β₀ rose by 1"), ((0, 1), "β₁ rose by 1"), ((0, -1), "β₁ dropped by 1"),
                                      ((-2, 0), "β₀ dropped by 2")])
def test_each_delta_beta_sign(d, needle):
    t = explain_verifier("X", R(d, (5, 2), (5 + d[0], 2 + d[1])))
    assert t[0].startswith(needle)


def test_near_miss_pass_and_fail_branches_match_the_rule():
    t = " ".join(explain_verifier("NEAR_MISS", R((0, 0), (10, 0), (10, 0))))
    assert "all hold → NEAR_MISS" in t and "bottleneck distance 0.500 ≥ d* 0.078" in t and "peak-z 26.1 ≥ z* 5.6" in t
    t = " ".join(explain_verifier("NUISANCE", R((0, 0), (10, 0), (10, 0), max_sdf_energy=1.2)))
    assert "max SDF energy 1.20 < e* 2.62 fails → NUISANCE" in t
    t = " ".join(explain_verifier("NUISANCE", R((0, 0), (10, 0), (10, 0), near_miss_enabled=False, d_star=None)))
    assert "near-miss rule is off" in t


def test_baseline_and_truth_sentences():
    assert "0.830 ≥ threshold 0.390" in explain_cnn({"prob": 0.83, "threshold": 0.39, "flagged": True, "domain_note": None})[0]
    c = explain_cnn({"prob": 0.1, "threshold": 0.39, "flagged": False, "domain_note": "outside its training domain"})
    assert "not a killer" in c[0] and "outside its training domain" in c[1]
    d = explain_d2d({"available": True, "single": {"flagged": False, "max_area": 31, "rule": {"theta": 10.0, "sigma": 1.0, "min_area": 48}},
                     "double": {"available": False, "why": "not available: double difference needs two reference dies"}})
    assert "covers 31 px < minimum area 48 px → not flagged" in d[0] and "needs two reference dies" in d[1]
    assert explain_d2d({"available": False, "why": "not available: no reference die"}) == ["Die-to-die: not available: no reference die."]
    assert "no verdict is marked correct or wrong" in explain_truth(None)[0]


def test_sentence_numbers_come_from_the_response(client):
    """Every number in the verifier and CNN sentences appears in the response fields it explains."""
    v = client.post("/api/v1/verify", json={"simulate": {"kind": "notch", "seed": 4}}).json()
    rs = v["verifier"]["reason"]
    allowed = set()
    for x in [*rs["delta_betti"], *rs["beta_ref"], *rs["beta_img"], rs["tau"]]:
        allowed |= {f"{abs(x):g}"}
    for k, nd in (("bottleneck_max", 3), ("d_star", 3), ("max_sdf_energy", 2), ("e_star", 2), ("peak_z", 1), ("z_star", 1)):
        allowed.add(f"{rs[k]:.{nd}f}")
    allowed |= {f"{abs(s):.2f}" for s in v["verifier"]["recovered_shift"] + v["truth"]["applied_shift"]}
    allowed |= {f"{v['cnn']['prob']:.3f}", f"{v['cnn']['threshold']:.3f}", "0", "1", "2"}
    text = " ".join(v["explanation"]["verifier"] + v["explanation"]["cnn"]).replace("β₀", "").replace("β₁", "")
    nums = set(re.findall(r"\d+\.\d+|\d+", text)) - {"3"}   # '3' only as 'Module 3'
    assert nums <= allowed, nums - allowed


@pytest.mark.parametrize("family,kind,seed", [("manhattan", "bridge", 0), ("curvy", "open", 1), ("via", "particle", 0), ("manhattan", "open", 1)])
def test_locator_sites_overlap_the_true_defect(client, family, kind, seed):
    """On simulated killers the verifier calls correctly, the marked site overlaps the injected defect mask."""
    v = client.post("/api/v1/verify", json={"simulate": {"family": family, "kind": kind, "seed": seed}}).json()
    assert v["verifier"]["verdict"].startswith("KILLER") and v["locator"]["agrees_with_delta_beta"] is True
    from nanovate.app.sources import SimulatedSource
    mask = SimulatedSource(family, kind, seed=seed).load().truth.mask
    hit = any(mask[max(r0 - 3, 0):r1 + 3, max(c0 - 3, 0):c1 + 3].any() for r0, c0, r1, c1 in (s["box"] for s in v["locator"]["sites"]))
    assert hit


def test_locator_marks_nothing_when_inconsistent_or_unchanged():
    win = Window(256, 16)
    cad = np.zeros((256, 256), bool); cad[100:120, :] = True; cad[140:160, :] = True
    field = cad.astype(float)                                  # identical to CAD: no difference blob
    e = np.zeros((224, 224)); e[50, 60] = 3.0
    out = locate(field, cad, (-1, 0), e, win)                  # Δβ claims a bridge the field does not show
    assert out["sites"] == [] and out["agrees_with_delta_beta"] is False
    assert out["energy_peak"] == {"row": 66, "col": 76, "value": 3.0}
    out = locate(field, cad, (0, 0), e, win)
    assert out["sites"] == [] and out["agrees_with_delta_beta"] is None and out["applicable"] is False
    bridged = field.copy(); bridged[120:140, 50:58] = 1.0     # a real join between the two wires
    out = locate(bridged, cad, (-1, 0), e, win)
    assert out["agrees_with_delta_beta"] is True and out["sites"][0]["type"] == "join" and out["sites"][0]["box"] == [120, 50, 140, 58]


def test_overlays_are_returned(client):
    v = client.post("/api/v1/verify", json={"simulate": {"kind": "bridge"}, "include_arrays": True}).json()
    assert v["overlays"]["cad_outline"]["png"].startswith("data:image/png;base64,")
    assert v["overlays"]["sdf_energy"]["offset"] == 16 and decode_array(v["arrays"]["sdf_energy"]).shape == (224, 224)
