"""M2 real-wiring test helpers: SVG rasteriser, label-completeness QC, and the run path on synthetic arrays."""
import numpy as np
from scipy import ndimage as ndi

from nanovate.real.m2 import rasterize_svg, unlabelled_conductor

SVG = ('<svg width="64" height="48"><path d="M2,10 L60,10 L60,16 L2,16 Z" fill="lime" />'
       '<circle cx="30" cy="35" r="4" fill="red" /></svg>')


def test_rasterize_paths_and_circles():
    r = rasterize_svg(SVG, (48, 64))
    assert r[12, 30] and r[35, 30] and not r[25, 30] and not r[12, 63]
    assert 59 * 7 + 40 < r.sum() < 59 * 7 + 60      # filled rectangle incl. boundary + a radius-4 disk (~49 px)


def _image(lab, extra=None):
    img = np.where(lab, 120.0, 40.0)
    if extra is not None:
        img = np.where(extra, 120.0, img)
    return ndi.gaussian_filter(img, 1.0) + np.random.default_rng(0).normal(0, 3, lab.shape)


def test_qc_flags_an_unlabelled_track_and_not_a_labelled_one():
    lab = np.zeros((200, 300), bool); lab[40:55, 10:290] = True
    miss = np.zeros_like(lab); miss[120:135, 10:290] = True
    assert not unlabelled_conductor(_image(lab), lab).any()
    q = unlabelled_conductor(_image(lab, miss), lab)
    assert q[127, 150] and not q[47, 150]


def test_run_path_smoke(monkeypatch):
    import experiments.real_m2 as M
    lab = np.zeros((512, 768), bool)
    for y in range(30, 512, 90):
        lab[y:y + 25, 20:740] = True
    img = _image(lab)
    monkeypatch.setattr(M, "arm_arrays", lambda i, s: (img, lab, np.zeros_like(lab)))
    monkeypatch.setattr(M, "ids", lambda: [f"{k:04d}" for k in range(321)])
    M._CAL.clear()
    rows = M._image_task(dict(arm="native", id="0001"))
    assert len(rows) == 6 and all(r.get("verdict") or r.get("skipped") for r in rows)
    import pandas as pd
    s = M.summarise(pd.DataFrame(rows))
    assert {"KILLER", "NEAR_MISS", "coverage"} <= set(s.metric)
