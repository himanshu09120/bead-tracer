"""
Table VII / Table X rows for the two single-run phi=0.80 models (evaluated by evaluate_model.py at phi=0.95):
  m163258        historical run, seed unrecorded
  phi080_s2026   re-run with recorded seed 2026
Held-out = mean over the 10 shapes of the per-shape mean coverage (20 starts each), +- sample SD (ddof=1) over shapes.
Writes results/table_x_extra.json.
"""
import json, sys
import numpy as np
from analysis.common import *

LABELS = {"m163258": "unrecorded", "phi080_s2026": "2026"}


def row(label, seed):
    g = lambda t: json.load(open(os.path.join(RAW, f"ppo_{label}__full__s{seed}__{t}__test.json")))
    out = {}
    for t in ["path1", "path2", "path3"]:
        r = g(t)
        out[t] = dict(r["summary"], per_start=[(e["coverage_pct"], e["complete"], round(e["ret"], 1), e["steps"], e["end_reason"]) for e in r["episodes"]])
    shapes = [g(t)["summary"]["coverage_mean"] for t in HELDOUT]
    out["heldout_shape_means"] = shapes
    out["heldout_mean"] = float(np.mean(shapes))
    out["heldout_sd_ddof1"] = float(np.std(shapes, ddof=1))
    out["heldout_pooled_episode_mean"] = float(np.mean([e["coverage_pct"] for t in HELDOUT for e in g(t)["episodes"]]))
    return out


if __name__ == "__main__":
    res = {}
    for label, seed in LABELS.items():
        try:
            res[label] = row(label, seed)
        except FileNotFoundError as e:
            print("missing", e); continue
        p = res[label]["path1"]
        print(f"{label}: path1 {p['coverage_mean']:.1f} +- {p['coverage_sd']:.1f}  {p['n_complete']}/20  ret {p['ret_mean']:.1f}  steps {p['steps_mean']:.0f}"
              f" | path2 {res[label]['path2']['coverage_mean']:.1f} path3 {res[label]['path3']['coverage_mean']:.1f}"
              f" | heldout {res[label]['heldout_mean']:.1f} +- {res[label]['heldout_sd_ddof1']:.1f} (pooled {res[label]['heldout_pooled_episode_mean']:.1f})")
    json.dump(res, open(os.path.join(ROOT, "results", "table_x_extra.json"), "w"), indent=1)
