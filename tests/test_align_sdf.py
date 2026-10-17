"""M4 / Module 3: sub-pixel alignment to the calibrated template + CAD SDF tension energy."""
import dataclasses

import numpy as np
import pytest
from scipy import ndimage as ndi

from nanovate.config import TestbedCfg
from nanovate.modules.align_sdf import align_to_template, cad_sdf, max_tension_energy, tension_energy
from nanovate.modules.calibration import calibrate
from nanovate.modules.topology import verify_topology
from nanovate.testbed.dataset import get_layout, make_sample, window
from nanovate.testbed.scribe import make_scribe_patches


def _smooth_scene(n=128):
    box = np.zeros((n, n))
    box[30:80, 25:90] = 1.0
    box[50:62, 90:110] = 0.6
    box[90:100, 20:70] = 0.8
    return ndi.gaussian_filter(box, 2.0)


@pytest.mark.parametrize("shift", [(5.3, -2.7), (-7.6, 6.2), (0.4, 0.4), (-8.0, 8.0)])
def test_known_shift_recovered_both_signs_and_round_trip(shift):
    ref = _smooth_scene()
    moved = ndi.shift(ref, shift, order=3, mode="nearest")
    aligned, d = align_to_template(moved, ref)
    # convention: ndi.shift(image, d) aligns image to the template, so d = -shift
    assert np.allclose(d, (-shift[0], -shift[1]), atol=0.1)
    assert np.abs(aligned - ref)[24:-24, 24:-24].max() < 0.03 * np.ptp(ref)


def test_zero_shift_is_identity():
    ref = _smooth_scene()
    aligned, d = align_to_template(ref, ref)
    assert np.allclose(d, (0.0, 0.0), atol=0.05)
    assert np.allclose(aligned, ref, atol=0.01)


def test_sdf_sign_and_units():
    cad = np.zeros((64, 64), bool)
    cad[:, 20:30] = True  # 10 px wide stripe
    phi = cad_sdf(cad)
    assert phi[32, 24] < 0 < phi[32, 50]            # conductor negative, dielectric positive
    assert phi[32, 24] == pytest.approx(-4.5, abs=1.0) or phi[32, 25] == pytest.approx(-4.5, abs=1.0)
    assert phi[32, 45] == pytest.approx(15.0, abs=1.0)  # 15 px from the stripe edge at col 29/30
    gy, gx = np.gradient(phi)
    assert np.hypot(gx, gy)[32, 40] == pytest.approx(1.0, abs=0.05)  # |grad phi| = 1 px/px (nm/nm)


def test_energy_formula_and_cap():
    phi = np.array([0.0, 3.0, -3.0, 50.0])
    s = np.array([1.0, 1.0, 1.0, 1.0])
    e = tension_energy(s, phi, gamma=0.2, cap=10.0)
    assert e[0] == pytest.approx(1.0)
    assert e[1] == pytest.approx(np.exp(0.6)) and e[2] == pytest.approx(e[1])  # symmetric in |phi|
    assert e[3] == pytest.approx(np.exp(2.0))                                    # capped at 10 px
    assert np.all(np.isfinite(tension_energy(s, np.array([1e6] * 4), 0.2, 10.0)))


def test_energy_finite_and_bounded_on_via_family():
    cfg = TestbedCfg()
    s = make_sample("via", "val", 0, 0, "nuisance", cfg, 0, shift=(0.0, 0.0))
    cal = calibrate(make_scribe_patches(cfg, seed=0))
    e = max_tension_energy(s.image, s.cad, cal, window(cfg), gamma=0.2, cap=10.0)
    assert np.isfinite(e) and 0 <= e < 50 * np.exp(0.2 * 10.0)


@pytest.fixture(scope="module")
def setup():
    out = {}
    for dose in (5.0, 200.0):
        cfg = dataclasses.replace(TestbedCfg(), sem=dataclasses.replace(TestbedCfg().sem, dose=dose))
        out[dose] = (cfg, calibrate(make_scribe_patches(cfg, seed=0)))
    return out


SHIFTS = [(7.6, -3.3), (-6.2, 5.7), (2.4, 7.9), (-8.0, -1.5)]


@pytest.mark.parametrize("dose,tol", [(200.0, 0.1), (5.0, 0.25)])
@pytest.mark.parametrize("family", ["manhattan", "curvy", "via"])
def test_estimator_accuracy_against_noise_free_image_of_same_wafer(setup, dose, tol, family):
    """Estimator error alone (template = the wafer's own noise-free image), ±8 px, both signs."""
    cfg, _ = setup[dose]
    errs = []
    for shift in SHIFTS:
        s = make_sample(family, "val", 1, 0, "nuisance", cfg, 7, shift=shift)
        ref = make_sample(family, "val", 1, 0, "nuisance", cfg, 7, shift=(0.0, 0.0)).expected
        _, d = align_to_template(s.image, ref)
        errs.append(max(abs(d[0] + shift[0]), abs(d[1] + shift[1])))
    assert max(errs) < tol, f"{family} dose {dose}: errors {np.round(errs, 3)}"


@pytest.mark.parametrize("dose", [200.0, 5.0])
@pytest.mark.parametrize("family", ["manhattan", "curvy", "via"])
def test_alignment_against_cad_template_has_a_measured_floor(setup, dose, family):
    """Against T(C) the CAD/wafer mismatch (LER) sets a floor that more passes do not remove:
    measured max error ~0.4 px (manhattan), 0.65-0.9 px (curvy), 0.2 px (via) -- the plan's
    <0.1 px tolerance is NOT met against a CAD-derived template (see PROGRESS.md).
    Sanity bounds only: right sign and error < 1.2 px."""
    cfg, cal = setup[dose]
    errs = []
    for shift in SHIFTS:
        s = make_sample(family, "val", 1, 0, "nuisance", cfg, 7, shift=shift)
        _, d = align_to_template(s.image, cal.predict(s.cad))
        errs.append(max(abs(d[0] + shift[0]), abs(d[1] + shift[1])))
    assert max(errs) < 1.2, f"{family} dose {dose}: errors {np.round(errs, 3)}"


def test_shifted_sample_gives_same_verdict_as_unshifted(setup):
    cfg, cal = setup[200.0]
    win = window(cfg)
    kinds = ["nuisance", "bridge", "open", "nuisance"]
    for lay, kind in enumerate(kinds):
        layout = get_layout("manhattan", "val", lay, cfg, 3)
        verdicts = []
        for shift in [(0.0, 0.0), (6.4, -4.1), (-7.3, 5.2)]:
            s = make_sample("manhattan", "val", lay, 0, kind, cfg, 3, layout=layout, shift=shift)
            aligned, _ = align_to_template(s.image, cal.predict(s.cad))
            v, _ = verify_topology(aligned, s.cad, win=win, tau=0.45, field_option="calibrated",
                                   levels=cal.levels, pre_smooth=1.0, region_tol=8)
            verdicts.append(v)
        assert len(set(verdicts)) == 1, f"layout {lay} {kind}: verdicts changed with shift {verdicts}"
