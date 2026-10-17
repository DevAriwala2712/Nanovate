"""M8: the demo walk-through module and the notebook that calls it (executed headlessly, no jupyter)."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from nanovate.demo import describe, load_frozen_cfg, plot_walk_through, walk_through
from nanovate.pipeline import PipelineCfg

NB = Path(__file__).resolve().parents[1] / "notebooks" / "demo.ipynb"


def test_walk_through_bridge_exposes_every_stage():
    w = walk_through("bridge", dose=200.0, layout_id=0, split="val", shift=(3.2, -2.1), pc=PipelineCfg())
    assert w.result.verdict == "KILLER_BRIDGE" and w.sample.label == "KILLER"
    assert w.sample.cad.shape == w.aligned.shape == w.field.shape == w.template.shape == (256, 256)
    assert 0.0 <= w.field.min() and w.field.max() <= 1.0
    assert max(abs(w.shift_est[0] + 3.2), abs(w.shift_est[1] - 2.1)) < 1.0   # recovered shift ~ -(applied)
    assert w.cal.mu_cond > w.cal.mu_diel and w.energy_map.shape == (256, 256)
    for k in ("diag_img_h0", "diag_img_h1", "diag_cad_h0", "diag_cad_h1"):
        assert getattr(w, k).ndim == 2 and getattr(w, k).shape[1] == 2


def test_describe_states_verdict_reason_and_singleton_flag():
    w = walk_through("nuisance", dose=200.0, layout_id=0, split="val", shift=(0.0, 0.0), pc=PipelineCfg())
    t = describe(w)
    assert w.result.verdict in t and "singleton" in t.lower() and "delta" in t.lower() and "synthetic" in t.lower()
    assert "SDF" in t and "bottleneck" in t.lower()


def test_plot_returns_a_figure_with_all_panels(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    w = walk_through("open", dose=200.0, layout_id=1, split="val", shift=(1.0, 1.0), pc=PipelineCfg())
    fig = plot_walk_through(w)
    assert len(fig.axes) >= 9
    fig.savefig(tmp_path / "walk.png")
    assert (tmp_path / "walk.png").stat().st_size > 10_000


def test_load_frozen_cfg_reads_decisions_and_falls_back(tmp_path):
    assert load_frozen_cfg(tmp_path / "missing.json") == PipelineCfg()
    p = tmp_path / "d.json"
    p.write_text(json.dumps({"pipeline_cfg": {**PipelineCfg().to_dict(), "tau": 0.4, "d_star": None, "e_star": None}}))
    c = load_frozen_cfg(p)
    assert c.tau == 0.4 and np.isinf(c.d_star) and np.isinf(c.e_star)


def test_notebook_is_valid_and_every_code_cell_executes(monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    nb = json.loads(NB.read_text())
    assert nb["nbformat"] == 4 and len(nb["cells"]) >= 8
    assert any("synthetic" in "".join(c["source"]).lower() for c in nb["cells"])
    ns: dict = {"__name__": "__notebook__"}
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            src = "".join(c["source"])
            exec(compile(src, f"demo.ipynb[cell {i}]", "exec"), ns)
    assert "walks" in ns and len(ns["walks"]) >= 3
    verdicts = {k: w.result.verdict for k, w in ns["walks"].items()}
    assert verdicts["bridge"] == "KILLER_BRIDGE"
