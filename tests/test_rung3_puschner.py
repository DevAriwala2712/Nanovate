"""Rung 3 harness on synthetic windows (no real data): verify-tile task and summary."""
import pickle

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

import experiments.rung3_puschner as R
from nanovate.pipeline_robust import RobustCfg
from nanovate.refcell import build_cell_reference


def _cell(swap=False):
    m = np.zeros((256, 256), bool)
    for c in (60, 88, 116, 144, 172):
        m[70:186, c:c + 14] = True
    if swap:
        m[120:134, 116:144] = True
    return ndi.gaussian_filter(np.where(m, 140.0, 40.0), 1.5) + np.random.default_rng(int(swap)).normal(0, 4, m.shape)


def test_verify_tile_task_and_summary(tmp_path, monkeypatch):
    _, _, rcfg = R.frozen()
    ref = build_cell_reference([_cell() for _ in range(5)], rcfg)
    monkeypatch.setattr(R, "WORK", tmp_path)
    pickle.dump({(("T1", "0"), 0): ref}, open(tmp_path / "references.pkl", "wb"))
    R._REFS.clear()
    ims = {0: _cell(), 1: _cell(swap=True)}
    monkeypatch.setattr(R, "load_window", lambda z, f, w, cache=None: ims[int(w["i"])])
    monkeypatch.setattr(R.zipfile, "ZipFile", lambda *a, **k: __import__("contextlib").nullcontext())
    rows = [dict(i=i, j=0, key=("T1", "0"), box_r0=64, box_r1=192, box_c0=48, box_c1=208, win_r0=0, win_c0=0) for i in (0, 1)]
    rows.append(dict(i=2, j=0, key=("T2", "0"), box_r0=0, box_r1=1, box_c0=0, box_c1=1, win_r0=0, win_c0=0))
    out = pd.DataFrame(R._verify_tile_task(("tileA", rows)))
    assert out.verdict.tolist()[2] == "NO_REFERENCE"
    assert not out.verdict[0].startswith("KILLER") and out.verdict[1].startswith("KILLER")
    inst = pd.DataFrame(dict(cell_type=["T1", "T1", "T2"], orient="0", filler=False, x0=0.0, y0=0.0, x1=1.0, y1=1.0))
    s, agg = R.summarise(out, inst)
    assert agg.loc[1, "killer"] and not agg.loc[0, "killer"]
    assert {"KILLER (any window)", "coverage = verified instances / all GDS instances"} <= set(s.metric)
