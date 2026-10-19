"""Track C Rung 1 helpers: fitted lattice reference, registration, tiling, run guard. Synthetic arrays only."""
import numpy as np
import pytest
from scipy import ndimage as ndi

from nanovate.real.lattice import Lattice, detect_holes, render
from nanovate.real.rung1 import (TILE, best_cal_tile, design_diameter, fit_basis_nm, holes_inside, register_origin,
                                 tile_grid)
from nanovate.topo_utils import betti_binary


def _centres(a1, a2, o, n=4):
    return np.array([np.add(o, i * np.asarray(a1) + j * np.asarray(a2)) for i in range(n) for j in range(n)])


def test_render_counts_holes_as_beta1():
    lat = Lattice((0.0, 60.0), (60.0, 0.0), (30.0, 30.0), 20.0)
    cad = render(lat, (240, 240))
    assert betti_binary(cad) == (1, 16)          # one film, 4 x 4 holes fully inside
    assert holes_inside(cad) == 16


def test_render_hexagonal_lattice():
    lat = Lattice((0.0, 50.0), (43.30127, 25.0), (100.0, 100.0), 16.0)
    cad = render(lat, (200, 200))
    b0, b1 = betti_binary(cad)
    assert b0 == 1 and b1 == holes_inside(cad) > 10


def test_fit_basis_joint_recovers_square_lattice():
    a1, a2 = np.array([0.4, 1515.0]), np.array([1520.0, -0.3])
    rng = np.random.default_rng(0)
    cs = [_centres(a1, a2, rng.uniform(0, 1500, 2), 3) + rng.normal(0, 3, (9, 2)) for _ in range(3)]
    b1, b2 = fit_basis_nm(cs)
    got = sorted([b1, b2], key=lambda v: abs(v[1]), reverse=True)
    assert np.allclose(np.abs(got[0]), np.abs(a1), atol=3) and np.allclose(np.abs(got[1]), np.abs(a2), atol=3)


def test_register_origin_recovers_offset_with_missing_and_extra_hole():
    a1, a2 = np.array([0.0, 100.0]), np.array([100.0, 0.0])
    o = np.array([37.0, 12.5])
    c = _centres(a1, a2, o, 4)
    c = np.delete(c, 5, axis=0)                   # a missing hole
    c = np.vstack([c, [[80.0, 80.0]]])            # an extra one
    got, res = register_origin(c, a1, a2)
    d = (got - o) / 100.0
    assert np.allclose(d - np.rint(d), 0, atol=1e-6)
    assert np.sum(res > 1e-6) == 1                # only the extra hole is off-lattice


def test_registered_reference_matches_detected_holes_on_rendered_image():
    lat = Lattice((0.0, 90.0), (90.0, 0.0), (41.0, 23.0), 30.0)
    img = np.where(render(lat, (512, 768)), 140.0, 20.0)
    img = ndi.gaussian_filter(img, 1.5) + np.random.default_rng(1).normal(0, 4, img.shape)
    c = np.array([r.centroid for r in detect_holes(img)])
    o, res = register_origin(c, np.array(lat.a1), np.array(lat.a2))
    assert np.median(res) < 0.5
    d = (o - np.array(lat.o)) / 90.0
    assert np.allclose(d - np.rint(d), 0, atol=0.02)


def test_tile_grid_and_cal_tile_rule():
    assert tile_grid((512, 768)) == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    assert tile_grid((300, 255)) == []
    cad = np.ones((512, 512), bool)
    cad[300:310, 300:310] = False
    cad[400:410, 400:410] = False
    cad[20:30, 20:30] = False
    assert best_cal_tile(cad) == (1, 1)
    assert TILE == 256


def test_design_diameter_parsing():
    assert design_diameter("2022_03_15_zset_120_dset_226_061.tif") == 226.0
    assert design_diameter("z50_d_100_6.tif") == 100.0
    assert design_diameter("z80_d_220-280_18.tif") is None
    assert design_diameter("z_110_overview_002.tif") is None


def test_lattice_scaled():
    lat = Lattice((0.0, 100.0), (100.0, 0.0), (10.0, 20.0), 30.0)
    s = lat.scaled(0.5)
    assert s.a1 == (0.0, 50.0) and s.o == (5.0, 10.0) and s.d == 15.0


def test_run_guard_refuses_without_committed_prereg(tmp_path, monkeypatch):
    from experiments import real_rung1
    monkeypatch.setattr(real_rung1, "_git", lambda *a: "")
    with pytest.raises(SystemExit):
        real_rung1.check_prereg(tmp_path)
