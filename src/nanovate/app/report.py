"""Downloadable report of one analysis: JSON (machine-readable) and a self-contained HTML page.

Both are built from the same /verify response body (inputs as PNGs + sha256, provenance, config, config hash,
seed, commit, all three verdicts, reasons, explanations, timings). The HTML reuses the UI's CSS variables
(read from frontend/index.html) and escapes every string. Nothing is added that the analysis did not produce.
"""
from __future__ import annotations

import datetime as _dt
import html
import json
import re
from pathlib import Path

DISCLAIMER = ("Nanovate demonstrator, synthetic-data prototype. It analyses one patch at a time and does not scan wafers or "
              "control any tool. No method here is validated on real SEM images.")


def build_json(resp: dict, config: dict) -> dict:
    """Report body. ``runtime`` (timings, cache state, generation time) is the only part that may differ between runs."""
    body = {k: v for k, v in resp.items() if k not in ("arrays", "request_id", "runtime")}
    label = ("SYNTHETIC DATA: simulated input with ground truth from the testbed wafer W" if resp["truth"] is not None
             else "LOADED FROM FILE: no ground truth exists for this input")
    return {"report": "Nanovate analysis report", "label": label, "disclaimer": DISCLAIMER, "config": config, **body,
            "runtime": {**resp["runtime"], "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                        "request_id": resp["request_id"]}}


def _css_vars(index: Path) -> str:
    text = index.read_text() if index.exists() else ""
    m = re.search(r":root\s*\{(.*?)\}", text, re.S)
    return m.group(1) if m else ""


def _t(rows: list[tuple]) -> str:
    e = html.escape
    return "<table>" + "".join(f"<tr><th>{e(str(k))}</th><td>{e(json.dumps(v) if isinstance(v, (list, dict)) else str(v))}</td></tr>"
                               for k, v in rows) + "</table>"


def build_html(rep: dict, index: Path = Path("frontend/index.html")) -> str:
    e = html.escape
    v, c, d, t = rep["verifier"], rep["cnn"], rep["d2d"], rep["truth"]
    file_like = rep["source_type"] == "file"
    imgs = [("SEM image", rep["images"]["sem"]["png"]), ("CAD", rep["images"]["cad"]["png"]),
            ("Conductor field P", rep["overlays"]["field"]["png"]), ("SDF energy (central window)", rep["overlays"]["sdf_energy"]["png"])]
    if "wafer" in rep["images"]:
        imgs.append(("Wafer W (ground truth)", rep["images"]["wafer"]["png"]))
    fig = "".join(f'<figure><div class="badge {"file" if file_like else ""}">{e(rep["provenance"])}</div>'
                  f'<img src="{e(src)}" alt="{e(name)}"><figcaption>{e(name)}</figcaption></figure>' for name, src in imgs)
    if d.get("available"):
        d2d = f'single: {"FLAGGED" if d["single"]["flagged"] else "not flagged"} (area {d["single"]["max_area"]} px)'
        if "flagged" in d.get("double", {}):
            d2d += f'; double: {"FLAGGED" if d["double"]["flagged"] else "not flagged"} (area {d["double"]["max_area"]} px)'
    else:
        d2d = d["why"]
    ag = rep["agreement"]
    pv = rep.get("pipeline_version", "unknown")
    verdicts = [(f"Verifier (topology, no labels) [pipeline {pv}]", v["verdict"], ag["verifier"]),
                (f"CNN (Manhattan-trained) [pipeline {pv}]", f'{"KILLER" if c["flagged"] else "NOT KILLER"} (p = {c["prob"]:.3f}, threshold {c["threshold"]:.3f})', ag["cnn"]),
                (f"Die-to-die [pipeline {pv}]", d2d, ag.get("d2d_single", "not available"))]
    vt = "<table><tr><th>Method</th><th>Verdict</th><th>vs ground truth</th></tr>" + "".join(
        f"<tr><td>{e(m)}</td><td>{e(x)}</td><td>{e(a)}</td></tr>" for m, x, a in verdicts) + "</table>"
    names = {"verifier": "Verifier", "locator": "Location", "cnn": "CNN", "d2d": "Die-to-die", "truth": "Ground truth"}
    xp = "".join(f"<h3>{e(names.get(k, k))}</h3><ul>" + "".join(f"<li>{e(s)}</li>" for s in lines) + "</ul>"
                 for k, lines in rep["explanation"].items() if lines)
    reason = _t([(k, val) for k, val in v["reason"].items()])
    truth = _t([("label", t["label"]), ("kind", t["kind"]), ("delta beta", t["d_beta"]), ("source", t["source"])]) if t else \
        "<p>No ground truth exists for this input; no verdict is marked correct or wrong.</p>"
    repro = _t([("pipeline version", pv), ("input id", rep["input_id"]), ("seed", rep["seed"]), ("commit", rep["commit"]), ("config hash", rep["config_hash"]),
                ("api version", rep["api_version"]), ("calibration", rep["calibration"]), ("params", rep["params"])])
    timings = _t([(k, val) for k, val in rep["runtime"].items()])
    warn = "".join(f'<p class="warn">{e(w)}</p>' for w in rep["warnings"])
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>Nanovate report {e(rep["input_id"])}</title>
<style>
:root {{{_css_vars(index)}}}
body {{ background: var(--bg-deep); color: var(--text); font-family: var(--font-ui); font-size: 13px; margin: 0; padding: 20px 24px; }}
h1 {{ font-size: 16px; color: var(--text-bright); margin: 0 0 4px; }} h2 {{ font-size: 12px; text-transform: uppercase; letter-spacing: .04em; color: var(--text-dim); margin: 20px 0 6px; border-bottom: 1px solid var(--border); padding-bottom: 4px; }}
h3 {{ font-size: 12px; color: var(--text-bright); margin: 10px 0 2px; }}
table {{ border-collapse: collapse; font-family: var(--font-mono); font-size: 12px; }} th, td {{ border: 1px solid var(--border); padding: 3px 8px; text-align: left; vertical-align: top; overflow-wrap: anywhere; }}
th {{ background: var(--bg-header); color: var(--text-dim); font-weight: 600; }}
.label {{ display: inline-block; background: {"var(--warn)" if file_like else "var(--danger)"}; color: #fff; font-weight: 600; font-size: 11px; padding: 2px 8px; }}
.pv {{ display: inline-block; background: var(--accent-dim); color: #fff; font: 600 11px var(--font-mono); padding: 1px 6px; vertical-align: middle; }}
.warn {{ color: var(--warn); margin: 4px 0; }} .dim {{ color: var(--text-dim); }}
.figs {{ display: flex; flex-wrap: wrap; gap: 12px; }} figure {{ margin: 0; position: relative; }} figure img {{ width: 220px; height: 220px; image-rendering: pixelated; display: block; border: 1px solid var(--border); }}
figcaption {{ font-size: 11px; color: var(--text-dim); margin-top: 3px; }}
.badge {{ position: absolute; top: 1px; left: 1px; background: var(--danger); color: #fff; font: 600 10px var(--font-mono); padding: 1px 5px; }} .badge.file {{ background: var(--warn); }}
</style></head><body>
<h1>Nanovate analysis report <span class="pv">pipeline {e(pv)}</span></h1>
<div class="label">{e(rep["label"])}</div>
<p class="dim">{e(rep["disclaimer"])}</p>
{warn}
<h2>Inputs ({e(rep["provenance"])})</h2><div class="figs">{fig}</div>
<h2>Verdicts</h2>{vt}
<h2>Explanation</h2>{xp}
<h2>Ground truth</h2>{truth}
<h2>Verifier reason</h2>{reason}
<h2>Reproducibility</h2>{repro}
<h2>Frozen configuration</h2><pre class="dim">{e(json.dumps(rep["config"], indent=1))}</pre>
<h2>Timings and runtime (not reproducible)</h2>{timings}
</body></html>
"""
