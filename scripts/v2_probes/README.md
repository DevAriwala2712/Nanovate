# v2 design probes (Manhattan **val** only, synthetic data)

These scripts produced the val evidence behind the v2 fixes (docs/V2_FIXES.md). They never touch the
test split. Run from the repo root with `PYTHONPATH=. .venv/bin/python scripts/v2_probes/<script> <out_dir>`.

| Script | What it measures | Used for |
|---|---|---|
| `h1fill_val.py` | v1 vs `h1_fill="dielectric"` verdicts, LER sigma 1.0/1.5/2.0 nm, doses 20/50/200 | Fix 2a (splice-seam holes) |
| `nmz_val.py` | delta-beta, bottleneck, SDF energy and peak-z of nuisance/notch/filament across LER and grain | Fix 2b (near-miss rule) |
| `nm_select_sim.py <nmz_val.csv>` | v1-form vs peak-z near-miss thresholds selected at nominal LER; alarm and recall per val cell | Fix 2b |
| `nm_norm_probe.py` | energy normalised by patch / scribe edge-band RMS and q99 (**rejected**: over-corrects at low LER) | Fix 2b (what did not work) |
| `nm_energy_probe.py` | energy with the edge band masked out (**rejected**: no effect, maxima are deep inside features) | Fix 2b (what did not work) |
| `grain_bridge.py` | bridges at grain 0/15/30%: verdict and conductor probability along the bridge | Exp 1 grain finding (bridge recall 79% at grain 30%) |
| `grain_tau.py` | tau 0.35/0.45/0.55 at grain 0/15/30%: recall per killer class and near-miss false bridges (**rejected**: trades opens and filaments for dim bridges) | Exp 1 grain finding |
