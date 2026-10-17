"""M7 support: smooth warp for Exp 4, template-anchored field (option c), aligned-image energy."""
import dataclasses

import numpy as np
import pytest

from nanovate.config import TestbedCfg
from nanovate.modules.align_sdf import align_to_template, max_tension_energy
from nanovate.modules.calibration import calibrate, features
from nanovate.modules.fields import build_field
from nanovate.testbed.dataset import get_layout, make_sample, window
from nanovate.testbed.scribe import make_scribe_patches
from nanovate.testbed.sem import warp_field


def test_warp_field_is_bounded_and_smooth():
    dy, dx = warp_field((300, 300), 1.0, np.random.default_rng(0))
    assert dy.shape == (300, 300)
    assert max(np.abs(dy).max(), np.abs(dx).max()) <= 1.0 + 1e-9
    assert max(np.abs(dy).max(), np.abs(dx).max()) > 0.5          # not degenerate
    assert np.abs(np.diff(dy, axis=1)).max() < 0.05                # slowly varying (px per px)
    z = warp_field((50, 50), 0.0, np.random.default_rng(0))
    assert not np.any(z[0]) and not np.any(z[1])


def test_zero_warp_is_bit_identical_to_no_warp():
    cfg = TestbedCfg()
    a = make_sample("manhattan", "val", 0, 0, "bridge", cfg, 1, shift=(2.0, -1.0))
    b = make_sample("manhattan", "val", 0, 0, "bridge", cfg, 1, shift=(2.0, -1.0), warp_amp=0.0)
    assert np.array_equal(a.image, b.image) and np.array_equal(a.expected, b.expected)


def test_warp_changes_the_image_but_not_the_wafer():
    cfg = TestbedCfg()
    a = make_sample("manhattan", "val", 0, 0, "bridge", cfg, 1, shift=(0.0, 0.0))
    b = make_sample("manhattan", "val", 0, 0, "bridge", cfg, 1, shift=(0.0, 0.0), warp_amp=1.0)
    assert np.array_equal(a.W, b.W) and a.label == b.label and a.topo == b.topo
    assert not np.array_equal(a.expected, b.expected)
    assert np.abs(a.expected - b.expected).max() < 0.5 * np.ptp(a.expected)  # sub-pixel warp, not a reshuffle


@pytest.fixture(scope="module")
def world():
    base = TestbedCfg()
    cfg = dataclasses.replace(base, sem=dataclasses.replace(base.sem, dose=100.0))
    cal = calibrate(make_scribe_patches(cfg, seed=0))
    layout = get_layout("manhattan", "val", 1, cfg, 4)
    return cfg, cal, layout


def test_template_field_equals_smoothed_cad_when_image_matches_template(world):
    cfg, cal, layout = world
    T = cal.predict(layout.cad)
    cs = features(layout.cad, *cal.sigmas)[..., 1]
    P, meta = build_field(T, "template", template=T, cad_smooth=cs, contrast=cal.mu_cond - cal.mu_diel)
    assert np.allclose(P, np.clip(cs, 0, 1)) and meta["field_option"] == "template"
    # a residual of one full contrast unit at a pixel drives it to conductor
    img = T.copy()
    img[100, 100] += cal.mu_cond - cal.mu_diel
    P2, _ = build_field(img, "template", template=T, cad_smooth=cs, contrast=cal.mu_cond - cal.mu_diel)
    assert P2[100, 100] == pytest.approx(1.0)


def test_template_field_handles_contrast_inversion(world):
    cfg, cal, layout = world
    T = np.where(layout.cad, 10.0, 40.0)            # conductor darker
    cs = layout.cad.astype(float)
    img = T.copy()
    img[5, 5] = 10.0                                # dielectric pixel that looks like conductor
    P, _ = build_field(img, "template", template=T, cad_smooth=cs, contrast=10.0 - 40.0)
    if not layout.cad[5, 5]:
        assert P[5, 5] == pytest.approx(1.0)


def test_energy_with_precomputed_alignment_matches(world):
    cfg, cal, layout = world
    s = make_sample("manhattan", "val", 1, 0, "bridge", cfg, 4, layout=layout)
    win = window(cfg)
    aligned, _ = align_to_template(s.image, cal.predict(s.cad))
    e1 = max_tension_energy(s.image, s.cad, cal, win)
    e2 = max_tension_energy(s.image, s.cad, cal, win, aligned=aligned)
    assert e1 == pytest.approx(e2)
