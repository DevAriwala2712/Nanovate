"""v2 design probe (Manhattan val only, synthetic data); see scripts/v2_probes/README.md."""
import sys, dataclasses, numpy as np, pandas as pd
from experiments.harness import nuisance_cfg
from experiments.select_config import with_dose, load_pipeline_cfg
from experiments.common import ExpCfg
from nanovate.config import TestbedCfg
from nanovate.modules.calibration import calibrate
from nanovate.pipeline import verify_patch
from nanovate.testbed.dataset import KINDS, get_layout, make_sample, window
from nanovate.testbed.scribe import make_scribe_patches
pc1 = load_pipeline_cfg(ExpCfg.full()); pc2 = dataclasses.replace(pc1, h1_fill="dielectric")
base = TestbedCfg(); win = window(base); rows=[]
for dose in (20, 50, 200):
  for sig in (1.0, 1.5, 2.0):
    tc = with_dose(nuisance_cfg(base, sig, 0.15), dose); cal = calibrate(make_scribe_patches(tc, seed=0))
    for lay in range(8):
      L = get_layout("manhattan","val",lay,tc,0)
      for die in range(2):
        for kind in KINDS:
          s = make_sample("manhattan","val",lay,die,kind,tc,0,layout=L)
          for name, pc in (("v1", pc1), ("h1diel", pc2)):
            r = verify_patch(s.image, s.cad, cal, win, pc)
            rows.append(dict(dose=dose, sigma=sig, layout=lay, die=die, kind=kind, killer=s.killer, cfg=name, verdict=r.verdict,
                             d_b=tuple(r.reason["delta_betti"]), bott=r.reason["bottleneck_max"], energy=r.reason["max_sdf_energy"]))
    print(dose, sig, flush=True)
pd.DataFrame(rows).to_csv(f"{sys.argv[1]}/h1fill_val.csv", index=False)
