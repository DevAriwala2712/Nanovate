"""M3 / Module 1: scribe-line ridge calibration + generalised Anscombe transform."""
import numpy as np
import pytest
from scipy import ndimage as ndi

from nanovate.config import TestbedCfg
from nanovate.modules.calibration import (
    Calibration,
    calibrate,
    estimate_shift,
    features,
    fit_noise_model,
    fit_noise_model_diff,
    gat,
    remove_ramp,
)
from nanovate.testbed.scribe import make_scribe_patches

S1, S2 = 1.5, 4.0


def _rect_cad(rng, n=96):
    c = np.zeros((n, n), bool)
    for _ in range(4):
        y, x = rng.integers(5, n - 40, 2)
        h, w = rng.integers(12, 36, 2)
        c[y:y + h, x:x + w] = True
    return c


def test_features_on_constant_fields():
    f1 = features(np.ones((32, 32)), S1, S2)
    f0 = features(np.zeros((32, 32)), S1, S2)
    assert f1.shape == (32, 32, 5)
    assert np.allclose(f1[8:24, 8:24].mean((0, 1)), [1, 1, 1, 0, 0], atol=1e-9)
    assert np.allclose(f0.mean((0, 1)), [1, 0, 0, 0, 0], atol=1e-9)


def test_recovers_known_weights_in_feature_span():
    rng = np.random.default_rng(0)
    w_true = np.array([12.0, 20.0, 3.0, 40.0, -15.0])
    patches = []
    for _ in range(3):
        cad = _rect_cad(rng)
        patches.append((cad, features(cad, S1, S2) @ w_true))
    cal = calibrate(patches, sigmas=(S1, S2), ridge=1e-9, register=False)
    assert np.allclose(cal.weights, w_true, rtol=1e-3)
    assert cal.mu_diel == pytest.approx(w_true[0], rel=1e-3)
    assert cal.mu_cond == pytest.approx(w_true[0] + w_true[1] + w_true[2], rel=1e-3)
    # the predicted template reproduces the training image
    cad, img = patches[0]
    assert np.allclose(cal.predict(cad), img, rtol=1e-3, atol=1e-2)


def test_contrast_inversion_is_handled():
    rng = np.random.default_rng(1)
    w_true = np.array([30.0, -18.0, -4.0, 25.0, 6.0])  # conductor darker than dielectric
    patches = [(c, features(c, S1, S2) @ w_true) for c in (_rect_cad(rng), _rect_cad(rng), _rect_cad(rng))]
    cal = calibrate(patches, sigmas=(S1, S2), ridge=1e-9, register=False)
    assert cal.mu_cond < cal.mu_diel
    assert cal.mu_cond == pytest.approx(30.0 - 22.0, rel=1e-3)


def test_noise_model_recovered_and_gat_stabilises_variance():
    rng = np.random.default_rng(2)
    alpha, s2 = 1.0, 1.0
    means = np.repeat([8.0, 15.0, 30.0, 60.0, 120.0, 240.0], 40_000)
    y = alpha * rng.poisson(means / alpha) + rng.normal(0, np.sqrt(s2), means.size)
    a_hat, s2_hat = fit_noise_model(means, y - means)
    assert a_hat == pytest.approx(alpha, rel=0.1)
    assert s2_hat == pytest.approx(s2, abs=1.5)
    for m in (15.0, 30.0, 60.0, 120.0, 240.0):
        yy = alpha * rng.poisson(m / alpha, 200_000) + rng.normal(0, np.sqrt(s2), 200_000)
        v = gat(yy, a_hat, s2_hat).var()
        assert 0.85 < v < 1.15, f"GAT variance {v:.3f} at mean {m}"


def test_gat_with_gain():
    rng = np.random.default_rng(3)
    alpha = 2.0
    yy = alpha * rng.poisson(40.0 / alpha, 200_000) + rng.normal(0, 1.0, 200_000)
    assert 0.85 < gat(yy, alpha, 1.0).var() < 1.15


def test_estimate_shift_recovers_known_shift_and_sign():
    box = np.zeros((128, 128))
    box[40:90, 35:100] = 1.0
    box[55:70, 100:115] = 0.6  # asymmetry so the peak is unique
    ref = ndi.gaussian_filter(box, 2.0)
    true = (2.3, -1.4)
    moved = ndi.shift(ref, true, order=3, mode="nearest")
    # estimate_shift(reference, moving) returns d with ndi.shift(moving, d) ~ reference
    d = estimate_shift(reference=ref, moving=moved)
    assert np.allclose(d, (-true[0], -true[1]), atol=0.1)
    back = ndi.shift(moved, d, order=3, mode="nearest")
    assert np.abs(back - ref)[20:-20, 20:-20].max() < 0.02 * np.ptp(ref)


def test_remove_ramp_keeps_levels():
    yy, xx = np.meshgrid(np.linspace(-1, 1, 64), np.linspace(-1, 1, 64), indexing="ij")
    template = np.where(xx > 0, 40.0, 10.0)
    ramp = 3.0 * xx + 2.0 * yy * yy - 1.0 * xx * yy
    ramp -= ramp.mean()
    out = remove_ramp(template + ramp, template)
    assert np.abs(out - template).max() < 0.5
    assert out.mean() == pytest.approx((template + ramp).mean(), abs=1e-6)


@pytest.fixture(scope="module")
def real_cal():
    cfg = TestbedCfg()
    return cfg, calibrate(make_scribe_patches(cfg, seed=0), sigmas=(S1, S2))


def test_calibration_on_scribe_patches_is_sane_and_timed(real_cal):
    cfg, cal = real_cal
    assert isinstance(cal, Calibration)
    assert cal.timing_ms > 0 and cal.timing_ms == cal.timing_ms  # recorded, finite
    assert cal.mu_cond > cal.mu_diel > 0
    # sanity band only (the transfer error is *measured* in scripts/, not asserted here)
    assert abs(cal.mu_diel / (cfg.sem.dose * cfg.sem.a0) - 1) < 0.3
    assert 0.6 < cal.alpha < 1.8


def test_neighbour_difference_noise_fit_ignores_ramp_and_grain():
    """Smooth charging ramp + blocky grain inflate a plain residual fit, not the difference fit."""
    rng = np.random.default_rng(5)
    n = 192
    yy, xx = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n), indexing="ij")
    template = np.where(xx > 0, 140.0, 50.0)
    grain = 1.0 + 0.25 * np.kron(rng.uniform(-1, 1, (12, 12)), np.ones((16, 16)))
    truth = template * np.where(xx > 0, grain, 1.0) + 25.0 * (yy + yy * yy - 1.0 / 3.0)  # zero-mean within each half, so the template mean stays right
    y = 1.0 * rng.poisson(truth) + rng.normal(0, 1.0, truth.shape)
    flat = np.ones(truth.shape, bool)
    a_plain, _ = fit_noise_model(template.ravel(), (y - template).ravel())
    a_diff, s2_diff = fit_noise_model_diff(y, template, flat)
    assert a_plain > 1.5, "setup: plain residual fit should be inflated by ramp/grain"
    assert a_diff == pytest.approx(1.0, rel=0.15)
    assert 0.0 <= s2_diff < 3.0


def test_moment_ridge_equals_design_matrix_ridge():
    """The 5x5-moment ridge solves the same normal equations as the explicit standardised design."""
    from nanovate.modules.calibration import _fit
    rng = np.random.default_rng(3)
    Fs = [features(_rect_cad(rng), S1, S2) for _ in range(3)]
    ys = [F @ np.array([2.0, 5.0, 1.0, -3.0, 0.5]) + rng.normal(0, 0.3, F.shape[:2]) for F in Fs]
    X = np.concatenate([f.reshape(-1, 5) for f in Fs]); y = np.concatenate([v.ravel() for v in ys])
    mu, sd = X[:, 1:].mean(0), X[:, 1:].std(0) + 1e-12
    Z = np.concatenate([np.ones((len(X), 1)), (X[:, 1:] - mu) / sd], axis=1)
    ws = np.linalg.solve(Z.T @ Z / len(Z) + np.diag([0, 1e-3, 1e-3, 1e-3, 1e-3]), Z.T @ y / len(Z))
    w_ref = np.r_[ws[0] - (ws[1:] / sd) @ mu, ws[1:] / sd]
    assert np.allclose(_fit(Fs, ys, 1e-3), w_ref, rtol=1e-8, atol=1e-8)


def test_shifted_features_equal_features_of_shifted_cad_in_interior():
    """Shift and blur commute: shifting G*C equals blurring the shifted CAD (away from the border)."""
    from nanovate.modules.calibration import _shifted_features
    cad = _rect_cad(np.random.default_rng(4)).astype(float)
    d = (0.37, -1.62)
    ref = features(ndi.shift(cad, d, order=1, mode="nearest"), S1, S2)
    got = _shifted_features(features(cad, S1, S2), d)
    inner = (slice(20, -20), slice(20, -20))
    assert np.abs(got[inner] - ref[inner]).max() < 1e-6
