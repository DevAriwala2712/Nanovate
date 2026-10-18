"""Part 1 (day-0 value case): break-even helpers, Colab training script smoke test, local evaluation."""
import json
import subprocess
import sys

import numpy as np
import pytest

from nanovate.baselines.breakeven import breakeven_subsets, finetune_epochs, pick_threshold_labelled


def test_subsets_sizes_disjoint_nested_and_seeded():
    ft, th = breakeven_subsets(768, 12, seed=0)
    assert len(ft) == 9 and len(th) == 3 and not set(ft) & set(th)
    ft2, th2 = breakeven_subsets(768, 12, seed=0)
    assert np.array_equal(ft, ft2) and np.array_equal(th, th2)
    assert not np.array_equal(breakeven_subsets(768, 12, seed=1)[0], ft)
    bft, bth = breakeven_subsets(768, 24, seed=0)
    assert set(np.r_[ft, th]) <= set(np.r_[bft, bth])          # nested in k for a seed
    assert all(len(a) == 0 for a in breakeven_subsets(768, 0, seed=0))
    with pytest.raises(ValueError):
        breakeven_subsets(10, 12, seed=0)


def test_threshold_fallbacks():
    assert pick_threshold_labelled(np.array([]), np.array([], bool)) == 0.39
    assert pick_threshold_labelled(np.array([0.2, 0.9]), np.array([False, False])) == 0.39
    assert pick_threshold_labelled(np.array([0.2, 0.6, 0.9]), np.array([False, True, True])) == 0.6


def test_finetune_epochs():
    assert finetune_epochs(0) == 0 and finetune_epochs(9) == 300 and finetune_epochs(576) == 17


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    """Tiny CPU run of the Colab script: via, k in {0, 12}, one seed, 2 optimiser steps."""
    out = tmp_path_factory.mktemp("be")
    subprocess.run([sys.executable, "scripts/breakeven_train.py", "--families", "via", "--k", "0,12",
                    "--seeds", "0", "--pool-layouts", "2", "--pool-dies", "1", "--steps", "2",
                    "--device", "cpu", "--out", str(out)], check=True)
    return out


def test_breakeven_train_smoke(smoke_run):
    m = json.loads((smoke_run / "manifest.json").read_text())
    runs = {(r["family"], r["k"]): r for r in m["runs"]}
    assert runs[("via", 0)]["weights"] is None
    assert (smoke_run / runs[("via", 12)]["weights"]).exists() and len(runs[("via", 12)]["thr_idx"]) == 3
    assert m["pool"] == {"layouts": 2, "dies": 1, "seed": 0}


def test_breakeven_k_helper():
    from experiments.breakeven import breakeven_k
    assert breakeven_k({0: 0.70, 12: 0.80, 24: 0.90}, 0.88) == 24
    assert breakeven_k({0: 0.70, 12: 0.80}, 0.88) == "none"
    assert breakeven_k({0: 0.90, 12: 0.95}, 0.88) == 0


def _tiny_cfg():
    from experiments.common import ExpCfg
    return ExpCfg(quick=True, layouts=1, dies=1, n_boot=20, doses=(50,), exp1_sigmas=(1.0,), exp1_grains=(0.0,), exp5_dies=1,
                  exp6_calib_runs=1, exp6_patches=1, val_layouts=1, val_dies=1)


def test_breakeven_eval_rows_k0_matches_m6_and_missing_weights(smoke_run, tmp_path, monkeypatch):
    import shutil
    from experiments import breakeven as be
    from nanovate.baselines.cnn import load_cnn, predict_proba
    from nanovate.config import TestbedCfg
    from nanovate.testbed.dataset import KINDS, make_sample
    from experiments.select_config import with_dose
    from experiments.common import f1_score
    monkeypatch.setenv("NANOVATE_RESULTS_ROOT", str(tmp_path))
    wdir = tmp_path / "w"; shutil.copytree(smoke_run, wdir)
    m = json.loads((wdir / "manifest.json").read_text())
    (wdir / m["runs"][1]["weights"]).unlink()                      # simulate a partial Colab run
    tab = be.run(_tiny_cfg(), wdir, verifier_csv="results/v2/exp3/geometry_f1.csv")
    k12 = tab[(tab.k == 12)]
    assert k12.missing.all() and k12.f1.isna().all()
    s = [make_sample("via", "test", 0, 0, kind, with_dose(TestbedCfg(), 50), 0) for kind in KINDS]
    model, _, _ = load_cnn("m6_cnn/cnn.pt")
    p = predict_proba(model, np.stack([x.image for x in s]), np.stack([x.cad for x in s]))
    k0 = tab[(tab.k == 0) & (tab.threshold_rule == "labelled")].iloc[0]
    assert k0.threshold == 0.39 and abs(k0.f1 - f1_score(p >= 0.39, np.array([x.killer for x in s]))) < 1e-12
    assert (tmp_path / "v2" / "quick" / "breakeven" / "summary.csv").exists()


def test_kaggle_kernel_dir_is_private_gpu_internet_and_pinned(tmp_path):
    """The generated Kaggle kernel clones the exact commit and runs the break-even script on the GPU."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("kp", "scripts/kaggle_breakeven.py"); kp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kp)
    d = kp.write_kernel(tmp_path / "k", user="someone", commit="abc1234def")
    meta = json.loads((d / "kernel-metadata.json").read_text())
    assert meta["id"] == "someone/nanovate-breakeven" and meta["is_private"] is True
    assert meta["enable_gpu"] is True and meta["enable_internet"] is True and meta["kernel_type"] == "notebook"
    src = "".join("".join(c["source"]) for c in json.loads((d / meta["code_file"]).read_text())["cells"])
    assert "git checkout -q abc1234def" in src and "scripts/breakeven_train.py --device auto --out /kaggle/working/breakeven" in src
