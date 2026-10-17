"""Regenerate every result: config selection (Manhattan val), experiments 1-6, and the README results table.

  python scripts/reproduce.py --quick     # small n, a few minutes, writes results/quick/ (README untouched)
  python scripts/reproduce.py             # full run, writes results/, rewrites the README table from the CSVs

One seed in ExpCfg drives everything; non-timing CSVs are bit-identical across runs on the same machine
(Exp 6 timings are not). The CNN is evaluated from m6_cnn/cnn.pt (trained off-machine by scripts/train_cnn.py).
Synthetic data throughout.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import exp1_nuisance, exp2_dose, exp3_geometry, exp4_misalignment, exp5_nearmiss, exp6_latency, report, select_config  # noqa: E402
from experiments.common import ExpCfg  # noqa: E402
from experiments.harness import Context  # noqa: E402

EXPERIMENTS = {"exp1": exp1_nuisance, "exp2": exp2_dose, "exp3": exp3_geometry, "exp4": exp4_misalignment, "exp5": exp5_nearmiss, "exp6": exp6_latency}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="small sample counts; results/quick/, README not touched")
    ap.add_argument("--skip-select", action="store_true", help="reuse results/[quick/]calib/decisions.json")
    ap.add_argument("--only", default="", help="comma-separated subset, e.g. exp2,exp5")
    ap.add_argument("--update-readme", action="store_true", help="rewrite the README table even for a --quick run")
    a = ap.parse_args()
    cfg = ExpCfg.quick() if a.quick else ExpCfg.full()
    root = select_config.results_root(cfg)
    t0 = time.time()
    if not a.skip_select:
        select_config.run(cfg)
    ctx = Context(cfg)
    chosen = [k for k in EXPERIMENTS if not a.only or k in a.only.split(",")]
    for name in chosen:
        t1 = time.time()
        print(f"== {name} ==", flush=True)
        EXPERIMENTS[name].run(cfg, ctx)
        print(f"   {name} done in {time.time() - t1:.0f}s", flush=True)
    if not a.quick or a.update_readme:
        report.update_readme("README.md", root)
        print("README results table rewritten from", root)
    else:
        print("quick run: README untouched; tables preview ->", root / "tables.md")
        (root / "tables.md").write_text(report.render_tables(root))
    print(f"all done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
