"""v2 design probe (Manhattan val only, synthetic data); see scripts/v2_probes/README.md."""
import sys, numpy as np, pandas as pd
d = pd.read_csv(sys.argv[1]); d["zero"] = d.d_b == "(0, 0)"; d["nm"] = d.kind != "nuisance"
d["killer"] = ~d.zero
sel = d[(d.sigma == 1.0) & (d.grain == 0.15)]   # pre-registered selection data: nominal nuisance, doses 20/50/200
nu = sel.kind == "nuisance"
def q(x): return np.unique(np.quantile(x, np.linspace(0, 0.98, 25)))
best = {}
for use_z in (False, True):
    zs = q(sel.z[nu & sel.zero]) if use_z else [-np.inf]
    cand = []
    for dq in q(sel.bott[nu & sel.zero]):
        for eq in q(sel.energy[nu & sel.zero]):
            for zq in zs:
                a = sel.zero & (sel.bott >= dq) & (sel.energy >= eq) & (sel.z >= zq)
                cand.append((a[nu].mean(), a[~nu].mean(), dq, eq, zq))
    C = pd.DataFrame(cand, columns=["fa","rec","d","e","z"]); C = C[C.fa <= 0.05 + 1e-12]
    b = C.sort_values(["rec","d","e","z"], ascending=False).iloc[0]; best[use_z] = b
    print("z" if use_z else "v1-form", f"d*={b.d:.3f} e*={b.e:.2f} z*={b.z:.2f} val-nominal FA={b.fa:.3f} rec={b.rec:.2f}")
    for (s, g), G in d.groupby(["sigma","grain"]):
        n = G.kind == "nuisance"
        nm_alarm = G.zero & (G.bott >= b.d) & (G.energy >= b.e) & (G.z >= b.z)
        supp = (~G.killer & ~nm_alarm)[n].mean()
        rec_any = (G.killer | nm_alarm)[~n].mean()
        print(f"   sigma {s} grain {g}: NM alarm on nuisance {nm_alarm[n].mean():.2f}  suppression {supp:.2f}  near-miss recall(any alarm) {rec_any:.2f}  (n_nu={n.sum()})")
