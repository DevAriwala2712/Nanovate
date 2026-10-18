"""v3 Phase 1 diagnostics: re-rendering at finer sampling and the instrumented verifier (synthetic data)."""
from __future__ import annotations

import numpy as np
import pytest

from nanovate.config import TestbedCfg
from nanovate.diagnostics import defect_saddle, delta_beta_curve, margin, saddle, scaled_window, verify_scaled
from nanovate.modules.calibration import calibrate
from nanovate.pipeline import PipelineCfg, verify_patch
from nanovate.testbed.dataset import make_sample, window
from nanovate.testbed.resample import make_fine_sample, make_fine_scribe_patches, upsample
from nanovate.testbed.scribe import make_scribe_patches
from nanovate.topo_utils import betti_binary

V2 = PipelineCfg(field_option="template", tau=0.35, pre_smooth=1.0, region_mode="splice", region_tol=8,
                 d_star=0.0775434990066642, e_star=2.6189942373880593, h1_fill="dielectric", z_star=5.578071347336651)
CFG = TestbedCfg()


@pytest.mark.parametrize("kind", ["bridge", "open", "nuisance"])
def test_fine_sample_at_s1_reproduces_testbed_expected_image(kind):
    a = make_sample("manhattan", "val", 0, 0, kind, CFG, seed=0)
    b = make_fine_sample("manhattan", "val", 0, 0, kind, CFG, seed=0, s=1)
    assert np.array_equal(a.expected, b.expected)
    assert np.array_equal(a.W, b.W) and np.array_equal(a.cad, b.cad)
    assert a.shift == b.shift_nm


def test_fine_scribe_at_s1_reproduces_testbed_scribe():
    for (c1, i1), (c2, i2) in zip(make_scribe_patches(CFG, seed=0), make_fine_scribe_patches(CFG, seed=0, s=1)):
        assert np.array_equal(c1, c2) and np.array_equal(i1, i2)


@pytest.mark.parametrize("s", [2, 3])
def test_upsampling_preserves_topology_and_physics(s):
    a = make_fine_sample("manhattan", "val", 1, 0, "bridge", CFG, seed=0, s=1)
    b = make_fine_sample("manhattan", "val", 1, 0, "bridge", CFG, seed=0, s=s)
    assert b.W.shape == (256 * s, 256 * s)
    assert betti_binary(a.W) == betti_binary(b.W) and betti_binary(a.cad) == betti_binary(b.cad)
    # same physical image: block-averaging the fine render recovers the coarse one closely (not exactly:
    # the blur acts on sub-pixels), and the conductor / dielectric plateaus agree
    coarse = b.expected.reshape(256, s, 256, s).mean((1, 3))
    assert np.median(np.abs(coarse - a.expected)) / a.expected.mean() < 0.02


def test_upsample_corner_touch_stays_unconnected():
    m = np.array([[1, 0], [0, 1]], bool)
    assert betti_binary(upsample(m, 3)) == betti_binary(m) == (2, 0)


@pytest.mark.parametrize("kind", ["bridge", "open", "particle", "nuisance", "filament"])
def test_verify_scaled_s1_equals_frozen_pipeline(kind):
    cal = calibrate(make_scribe_patches(CFG, seed=0))
    smp = make_sample("manhattan", "val", 2, 0, kind, CFG, seed=0)
    ref = verify_patch(smp.expected, smp.cad, cal, window(CFG), V2)
    got = verify_scaled(smp.expected, smp.cad, cal, window(CFG), V2, s=1)
    assert got.verdict == ref.verdict
    assert got.delta_beta == tuple(ref.reason["delta_betti"])
    assert got.bottleneck == pytest.approx(ref.reason["bottleneck_max"])
    assert got.energy == pytest.approx(ref.reason["max_sdf_energy"])
    curve = delta_beta_curve(got.diagrams, np.array([V2.tau]))
    assert tuple(curve[0]) == got.delta_beta


def test_scaled_window():
    w = scaled_window(window(CFG), 3)
    assert (w.size, w.margin) == (768, 48)


def test_delta_beta_curve_counts_lifetimes():
    d0 = np.array([[0.0, np.inf], [0.0, 0.5]])
    d1 = np.array([[0.2, 0.3]])
    ref = (np.array([[0.0, np.inf], [0.0, 1.0]]), np.empty((0, 2)))
    c = delta_beta_curve((d0, d1, *ref), np.array([0.05, 0.4, 0.6]))
    assert c.tolist() == [[0, 1], [0, 0], [-1, 0]]


def test_saddle_two_blocks_joined_by_dim_neck():
    P = np.zeros((5, 9))
    P[:, :3] = 1.0
    P[:, 6:] = 1.0
    P[2, 3:6] = [0.6, 0.4, 0.7]
    A = np.zeros_like(P, bool); A[:, 0] = True
    B = np.zeros_like(P, bool); B[:, -1] = True
    assert saddle(P, A, B, np.ones_like(A)) == pytest.approx(0.4)


def test_defect_saddle_bridge_and_margin_sign():
    smp = make_sample("manhattan", "val", 0, 0, "bridge", CFG, seed=0)
    sad = defect_saddle("bridge", smp.W.astype(float), smp.cad, smp.W, smp.mask, smp.params["defect"], CFG.layout.pad)
    assert sad["geom"] == "ok" and sad["saddle"] == 1.0     # on the binary wafer the bridge is full conductor
    assert margin("bridge", sad, 0.35) > 0
    smp = make_sample("manhattan", "val", 0, 0, "open", CFG, seed=0)
    sad = defect_saddle("open", smp.W.astype(float), smp.cad, smp.W, smp.mask, smp.params["defect"], CFG.layout.pad)
    assert sad["geom"] == "ok" and sad["saddle"] == 0.0     # the cut is dielectric all the way across
    assert margin("open", sad, 0.35) > 0


def test_separability_floor_and_mechanism():
    import pandas as pd
    from experiments.v3_phase1 import mechanism, separability
    df = pd.DataFrame(dict(kind=["open", "open", "notch", "notch"], saddle=[0.2, 0.7, 0.5, 0.9]))
    r = separability(df, "open", "notch", change_if_high=False)
    assert r["min_errors_any_threshold"] == 1 and r["auc"] == 0.75
    row = pd.Series(dict(kind="filament", killer=False, db0=1, db1=0, margin=-0.3))
    assert mechanism(row) == "filament_read_as_split"
    row = pd.Series(dict(kind="bridge", killer=True, db0=0, db1=0, margin=0.1))
    assert mechanism(row) == "defect_seen_but_cancelled_or_masked"
