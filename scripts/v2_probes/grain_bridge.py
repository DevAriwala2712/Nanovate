"""v2 design probe (Manhattan val only, synthetic data); see scripts/v2_probes/README.md."""
import sys, numpy as np, pandas as pd
from experiments.harness import nuisance_cfg
from experiments.select_config import with_dose, load_pipeline_cfg
from experiments.common import ExpCfg
from nanovate.config import TestbedCfg
from nanovate.modules.calibration import calibrate, features
from nanovate.modules.align_sdf import align_to_template
from nanovate.modules.fields import build_field
from nanovate.pipeline import verify_patch
from nanovate.testbed.dataset import get_layout, make_sample, window
from nanovate.testbed.scribe import make_scribe_patches
pc = load_pipeline_cfg(ExpCfg.full()); base = TestbedCfg(); win = window(base); rows = []
for gr in (0.0, 0.15, 0.30):
    tc = with_dose(nuisance_cfg(base, 1.0, gr), 50); cal = calibrate(make_scribe_patches(tc, seed=0))
    for lay in range(8):
        L = get_layout("manhattan", "val", lay, tc, 0); T = cal.predict(L.cad); cs = features(L.cad, *cal.sigmas)[..., 1]
        for die in range(4):
            s = make_sample("manhattan", "val", lay, die, "bridge", tc, 0, layout=L)
            r = verify_patch(s.image, s.cad, cal, win, pc)
            al, _ = align_to_template(s.image, T)
            f, _ = build_field(al, "template", levels=cal.levels, pre_smooth=1.0, template=T, cad_smooth=cs, contrast=cal.mu_cond - cal.mu_diel)
            fx, _ = build_field(s.expected, "template", levels=cal.levels, pre_smooth=0.0, template=T, cad_smooth=cs, contrast=cal.mu_cond - cal.mu_diel)
            m = s.mask
            rows.append(dict(grain=gr, layout=lay, die=die, verdict=r.verdict, d_b=tuple(r.reason["delta_betti"]),
                             radius=s.params.get("defect", {}).get("r", np.nan), area=int(m.sum()),
                             p_med=float(np.median(f[m])), p_min=float(np.min(f[m])), px_med=float(np.median(fx[m])),
                             grain_level=float(np.median((s.expected[m]))) / (cal.mu_cond)))
    print(gr, flush=True)
d = pd.DataFrame(rows); d.to_csv(f"{sys.argv[1]}/grain_bridge.csv", index=False)
d["hit"] = d.verdict.str.startswith("KILLER")
print(d.groupby("grain").agg(recall=("hit", "mean"), p_med=("p_med", "median"), px_med=("px_med", "median"), n=("hit", "size")))
print(d[~d.hit][["grain", "layout", "die", "verdict", "d_b", "radius", "area", "p_med", "p_min", "px_med", "grain_level"]].to_string())

