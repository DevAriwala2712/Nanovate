"""M1 testbed tests: geometry, nuisance statistics, defect labels, SEM units, determinism."""
import numpy as np
import pytest
from skimage.draw import polygon as sk_polygon

from nanovate.config import TestbedCfg
from nanovate.testbed.dataset import KINDS, make_sample, make_stack, window
from nanovate.testbed.labels import EXPECTED
from nanovate.testbed.layouts import FAMILIES, fill_polygon, min_space_ok
from nanovate.testbed.perturb import ou_process
from nanovate.testbed.scribe import make_scribe_patches
from nanovate.topo_utils import Window, betti_binary, betti_local, delta_betti, relevance_region


def test_fill_polygon_matches_skimage():
    rng = np.random.default_rng(0)
    N = 128
    for _ in range(100):
        n = rng.integers(3, 10)
        ang = np.sort(rng.uniform(0, 2 * np.pi, n))
        rad = rng.uniform(10, 70, n)
        pts = np.stack([64 + rad * np.cos(ang), 64 + rad * np.sin(ang)], 1) + rng.uniform(-0.4, 0.4, 2)
        ref = np.zeros((N, N), bool)
        rr, cc = sk_polygon(pts[:, 1], pts[:, 0], (N, N))
        ref[rr, cc] = True
        assert np.array_equal(ref, fill_polygon(pts, N))


def test_corner_touch_is_not_a_short():
    a = np.zeros((8, 8), bool)
    a[2, 2] = a[3, 3] = True
    assert betti_binary(a) == (2, 0)


def test_hole_counting():
    a = np.zeros((20, 20), bool)
    a[5:15, 5:15] = True
    a[8:12, 8:12] = False
    assert betti_binary(a) == (1, 1)
    # a hole closed only diagonally is still a hole connected to the outside (8-conn dielectric)
    b = a.copy()
    b[8:12, 8:12] = True
    b[9, 9] = False
    assert betti_binary(b) == (1, 1)


def test_margin_rule_ignores_slivers_in_margin():
    win = Window(64, 8)
    cad = np.zeros((64, 64), bool)
    cad[0:6, :] = True            # wire hugging the tile edge, entirely in the margin
    cad[30:36, :] = True          # wire through the centre
    w = cad.copy()
    w[0:6, ::4] = False           # wire in the margin fragments into many pieces
    assert betti_local(cad, win) == (1, 0)
    assert delta_betti(w, cad, win) == (0, 0)
    w2 = cad.copy()
    w2[30:36, 32:34] = False      # an open in the centre is counted
    assert delta_betti(w2, cad, win) == (1, 0)


def test_margin_rule_ignores_margin_holes():
    win = Window(64, 8)
    cad = np.ones((64, 64), bool)
    # create an enclosed dielectric hole sitting entirely in the margin (rows 1..5)
    cad[1:6, 10:20] = False
    w = cad.copy()
    # hole edge expands 1 px toward the central window
    w[1:7, 10:20] = False
    reg = relevance_region(cad, win)
    assert betti_local(cad, win, reg) == (1, 0)
    assert delta_betti(w, cad, win) == (0, 0)
    # whereas an enclosed hole in the centre is counted
    cad_inner = cad.copy()
    cad_inner[25:35, 25:35] = False
    reg_in = relevance_region(cad_inner, win)
    assert betti_local(cad_inner, win, reg_in) == (1, 1)


def test_ou_process_statistics():
    rng = np.random.default_rng(1)
    step, sigma, xi = 0.5, 1.5, 20.0
    e = np.concatenate([ou_process(4000, step, sigma, xi, rng) for _ in range(50)])
    assert abs(e.std() - sigma) < 0.1
    x = ou_process(400000, step, sigma, xi, rng)
    lag = int(xi / step)
    r = np.corrcoef(x[:-lag], x[lag:])[0, 1]
    assert abs(r - np.exp(-1)) < 0.05


@pytest.mark.parametrize("family", FAMILIES)
def test_layouts_are_drc_clean_and_deterministic(family):
    cfg = TestbedCfg()
    a = make_sample(family, "val", 3, 0, "nuisance", cfg, seed=7)
    b = make_sample(family, "val", 3, 0, "nuisance", cfg, seed=7)
    assert np.array_equal(a.cad, b.cad) and np.array_equal(a.W, b.W) and np.array_equal(a.image, b.image)
    assert min_space_ok(a.cad, cfg.layout.min_space)
    assert a.cad.shape == a.image.shape == (cfg.layout.tile,) * 2


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("kind", KINDS)
def test_labels_come_from_topology_of_W(family, kind):
    """Every class: stored label == topology recomputed from W vs CAD; class matches intent."""
    cfg = TestbedCfg()
    win = window(cfg)
    for die in range(3):
        s = make_sample(family, "val", 1, die, kind, cfg, seed=11)
        d = delta_betti(s.W, s.cad, win)
        assert d == (s.topo["d_beta0"], s.topo["d_beta1"])
        exp = EXPECTED[kind]
        if exp == "none":
            assert d == (0, 0)
        elif exp == "bridge":
            assert d[0] <= 0 and d[1] >= 0 and d != (0, 0)
        else:
            assert d[0] >= 0 and d[1] <= 0 and d != (0, 0)
        if kind != "nuisance":
            assert s.mask.any()
        # the LER-only wafer never changes topology inside a label (checked before injection)
        assert s.label == {"none": "NUISANCE" if kind == "nuisance" else "NEAR_MISS"}.get(exp, "KILLER")


def test_filament_stops_1_to_3_nm_short():
    cfg = TestbedCfg()
    for die in range(6):
        s = make_sample("manhattan", "val", 2, die, "filament", cfg, seed=3)
        gap = s.params["defect"]["gap"]
        assert 1.0 <= gap <= 3.75, gap


def test_defects_stay_out_of_margin():
    cfg = TestbedCfg()
    m = cfg.layout.margin
    for kind in ("bridge", "open", "particle", "notch", "filament"):
        s = make_sample("curvy", "val", 0, 0, kind, cfg, seed=5)
        rr, cc = np.nonzero(s.mask)
        assert rr.min() >= m and cc.min() >= m and rr.max() < 256 - m and cc.max() < 256 - m


def test_sem_units_and_shift():
    cfg = TestbedCfg()
    cfg.sem.dose = 100.0
    s = make_sample("via", "val", 0, 0, "nuisance", cfg, seed=2)
    # image is in e-/px: its mean matches the noise-free expectation (Poisson mean)
    assert abs(s.image.mean() - s.expected.mean()) < 0.02 * s.expected.mean()
    assert max(abs(v) for v in s.shift) <= cfg.sem.max_shift
    # noise variance tracks dose * mean (shot noise) + read noise
    resid = s.image - s.expected
    assert 0.8 < resid.var() / (s.expected.mean() + cfg.sem.read_noise**2) < 1.2


def test_stack_shares_cad_and_varies_nuisance():
    cfg = TestbedCfg()
    st = make_stack("manhattan", "val", 0, ["nuisance", "bridge", "nuisance"], cfg, seed=4)
    assert all(np.array_equal(st[0].cad, s.cad) for s in st)
    assert not np.array_equal(st[0].W, st[2].W)
    assert st[1].killer and not st[0].killer


def test_scribe_patches():
    cfg = TestbedCfg()
    sp = make_scribe_patches(cfg, seed=0)
    assert len(sp) == 3
    for cad, img in sp:
        assert cad.shape == img.shape == (256, 256) and 0.1 < cad.mean() < 0.9


def test_resample_counter_at_high_ler():
    """At sigma = 2.5 nm, LER topology changes occur and are tracked (AGENTS.md §7.6)."""
    import dataclasses
    cfg = TestbedCfg()
    c = dataclasses.replace(cfg, nuisance=dataclasses.replace(cfg.nuisance, ler_sigma=2.5))
    samples = [make_sample("manhattan", "val", i, 0, "nuisance", c, seed=0) for i in range(8)]
    total_resamples = sum(s.resamples for s in samples)
    total_ler_rejects = sum(s.params["resample_reasons"].get("ler_topology", 0) for s in samples)
    assert total_resamples > 0
    assert total_ler_rejects > 0

