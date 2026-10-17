"""Write notebooks/demo.ipynb (nbformat 4, no stored outputs). The notebook only calls nanovate.demo,
which is tested; tests/test_demo.py executes every code cell. Run: python scripts/make_demo_notebook.py"""
import json
from pathlib import Path


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


CELLS = [
    md("""
# Nanovate demo: one synthetic tile through every stage

**All data here is synthetic** (a simulated SEM testbed), and the tiles below are *illustrations* from the validation
split, not results; the measured results with baselines and confidence intervals are in the README and `results/`.

Pipeline: scribe-line **calibration** (Module 1) -> sub-pixel **alignment** to the calibrated template and **SDF energy**
(Module 3) -> conductor **field** -> **persistent homology** verdict against the CAD (Module 4). Every verdict returns its
reason: delta-beta, bottleneck distances and the maximum SDF energy.
"""),
    code("""
import matplotlib
matplotlib.use("Agg")          # headless-safe; in Jupyter use `%matplotlib inline` and drop this line
import matplotlib.pyplot as plt
from nanovate.demo import describe, plot_walk_through, walk_through
from nanovate.pipeline import verify_stack
from nanovate.testbed.dataset import get_layout, make_sample, window
from nanovate.config import TestbedCfg
"""),
    md("""
## 1. A bridge (short): the verdict comes from topology
We inject a conductive bridge, acquire the tile with a random sub-pixel shift, calibrate on three scribe patches, and run the pipeline.
"""),
    code("""
walks = {}
walks["bridge"] = walk_through("bridge", dose=100.0, layout_id=0, split="val", shift=(3.2, -2.1))
print(describe(walks["bridge"]))
fig = plot_walk_through(walks["bridge"])
"""),
    md("""
## 2. The other classes on the same layout
An open and a particle (both change the number of conductor components), a harmless nuisance sample (line-edge roughness only),
and a notch that does not sever the wire (near-miss). Delta-beta decides killer vs not; near-misses need the bottleneck + SDF-energy + peak-z rule (v2 config, results/v2/calib).
"""),
    code("""
for kind in ("open", "particle", "nuisance", "notch"):
    walks[kind] = walk_through(kind, dose=100.0, layout_id=1, split="val", shift=(-2.0, 1.5))
rows = [(k, w.sample.label, w.result.verdict, tuple(w.result.reason["delta_betti"]), round(w.result.reason["max_sdf_energy"], 2)) for k, w in walks.items()]
for r in rows:
    print("%-9s truth=%-9s verdict=%-22s delta-beta=%-9s max SDF energy=%s" % r)
"""),
    code("""
for kind in ("open", "nuisance"):
    print(describe(walks[kind]), "\\n")
fig2 = plot_walk_through(walks["open"])
"""),
    md("""
## 3. Singleton vs stack
With no same-layout peers the pipeline falls back to Gaussian pre-smoothing and flags the result `singleton=True`. With K same-layout dies,
Module 2 (rank-2 randomized SVD + MAD threshold, after aligning every die to the template) denoises each die and the flag is `False`.
"""),
    code("""
cfg = TestbedCfg()
layout = get_layout("manhattan", "val", 0, cfg, 0)
kinds = ["nuisance"] * 11 + ["bridge"]
dies = [make_sample("manhattan", "val", 0, k, kd, cfg, 0, layout=layout) for k, kd in enumerate(kinds)]
w = walks["bridge"]
stack = verify_stack([d.image for d in dies], layout.cad, w.cal, window(cfg), w.pc.__class__(**{**w.pc.to_dict(), "use_stack": True}))
print("stack of", len(stack), "dies; singleton flags:", sorted({r.singleton for r in stack}))
print("verdicts:", [r.verdict[:9] for r in stack])
print("single-patch path singleton flag:", walks["bridge"].result.singleton)
"""),
    md("""
## What to read next
`README.md` has the generated results table (verifier vs die-to-die vs a Manhattan-trained CNN, with CIs and n) and the limitations;
`PROGRESS.md` records every measurement, surprise and missed target.
"""),
]

NB = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "notebooks" / "demo.ipynb"
    out.write_text(json.dumps(NB, indent=1) + "\n")
    print("wrote", out)
