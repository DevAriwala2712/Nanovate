"""Regenerate every result: config selection (Manhattan val), experiments 1-6, and the README results table.

  python scripts/reproduce.py --quick     # small n, a few minutes, writes results/v2/quick/ (README untouched)
  python scripts/reproduce.py             # full v2 run, writes results/v2/, rewrites the README v2 table
  python scripts/reproduce.py --version v1   # the first, pre-registered run: results/, README v1 table
  python scripts/reproduce.py --version v3   # v3: event rule selected on val (results/v3/calib), Exp 1-6 on the fresh test seed
  python scripts/reproduce.py --version v3 --only day0   # v3 day-0 projection only (results/v3/day0/) from results/v3/exp3
  python scripts/reproduce.py --version v3 --only status # README status box only (v3 + Track C Rung 1 CSVs)

v1 is the pre-registered first run; v2 adds post-hoc fixes chosen on Manhattan val only (docs/V2_FIXES.md).

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

from experiments import breakeven, day0_scenario, exp1_nuisance, exp2_dose, exp3_geometry, exp4_misalignment, exp5_nearmiss, exp6_latency, report, select_config  # noqa: E402
from experiments.common import ExpCfg  # noqa: E402
from experiments.harness import Context  # noqa: E402

EXPERIMENTS = {"exp1": exp1_nuisance, "exp2": exp2_dose, "exp3": exp3_geometry, "exp4": exp4_misalignment, "exp5": exp5_nearmiss, "exp6": exp6_latency}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="small sample counts; results/quick/, README not touched")
    ap.add_argument("--skip-select", action="store_true", help="reuse results/[quick/]calib/decisions.json")
    ap.add_argument("--only", default="", help="comma-separated subset, e.g. exp2,exp5")
    ap.add_argument("--update-readme", action="store_true", help="rewrite the README table even for a --quick run")
    ap.add_argument("--version", default="v2", choices=("v1", "v2", "v3"), help="v1 = first pre-registered run, v2 = post-hoc fixes")
    ap.add_argument("--breakeven-weights", default="", help="dir with manifest.json + weights from notebooks/breakeven_colab.ipynb")
    a = ap.parse_args()
    if a.version == "v3":  # own selection + runner; v1 / v2 paths below are unchanged
        from experiments import v3_run, v3_select, v3_val_report
        only = set(a.only.split(",")) if a.only else set()
        if not only or only - {"day0", "status"}:  # --only day0 / status re-render from existing v3 results only
            if not a.skip_select:
                c = v3_select.V3Cfg.quick_cfg() if a.quick else v3_select.V3Cfg()
                v3_select.run(c)
                v3_val_report.run(c)
            v3_run.run(a.quick, ",".join(sorted(only - {"day0", "status"})))
        if not only or "day0" in only:
            cfg3 = ExpCfg.quick("v3") if a.quick else ExpCfg.full("v3")
            print("== day0 scenario (v3) ==", flush=True)
            day0_scenario.run(cfg3)
            if not a.quick or a.update_readme:
                day0 = report.render_day0(select_config.results_root(cfg3), day0_scenario.OUT_DIR["v3"], "v3")
                for doc in ("README.md", "docs/USE_CASE.md"):
                    report.update_block(doc, day0, (report.DAY0_V3_START, report.DAY0_V3_END))
                print("README / docs/USE_CASE.md v3 day-0 blocks updated")
        if (not a.quick or a.update_readme) and (not only or only & {"day0", "status"}):
            report.update_status("README.md")
            print("README status box updated")
        return
    cfg = ExpCfg.quick(a.version) if a.quick else ExpCfg.full(a.version)
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
    if a.version == "v2" and (not a.only or "day0" in a.only.split(",")):
        print("== day0 scenario ==", flush=True)
        day0_scenario.run(cfg)
        if a.breakeven_weights:
            print("== label break-even ==", flush=True)
            breakeven.run(cfg, a.breakeven_weights)
    if not a.quick or a.update_readme:
        if a.version == "v2" and (not a.only or set(a.only.split(",")) - {"day0"}):
            report.update_readme("README.md", root, (report.START, report.END))
        elif a.version == "v1":
            report.update_readme("README.md", root, (report.V1_START, report.V1_END))
        if a.version == "v2":
            day0 = report.render_day0(root, version="v2")
            for doc in ("README.md", "docs/USE_CASE.md"):
                report.update_block(doc, day0, (report.DAY0_START, report.DAY0_END))
        print("README / docs/USE_CASE.md generated blocks updated from", root)
    else:
        print("quick run: README untouched; tables preview ->", root / "tables.md")
        (root / "tables.md").write_text(report.render_tables(root))
    print(f"all done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
