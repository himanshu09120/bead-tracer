"""
Run every non-learned baseline.  Raw results -> results/raw/, sweep table -> results/sweep.json.

1. random / zero / raster : test starts on every target.
2. speed sweep (greedy, pure_pursuit, crop_greedy) over SPEEDS on path1 VALIDATION starts.
3. selected speed = fastest speed that completes all 20 validation starts (fallback, flagged in the json:
   the speed with most completions, fastest on ties).
4. test starts on every target at the selected speed AND at v_max (=15).

    python -m baselines.run_baselines [--targets path1 path2 path3 heldout00 ...]
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from evaluation import evaluate, save_result, result_path, make_env
from baselines import make_baseline

SPEEDS = [3, 4, 5, 6, 8, 10, 12]
VMAX = 15
SWEPT = ["greedy", "pure_pursuit", "crop_greedy"]


def run(kind, speed, target, split):
    pol = make_baseline(kind, speed)
    p = result_path(pol.name, "none", 0, target, split)
    if os.path.exists(p):
        return json.load(open(p))
    r = evaluate(pol, target, split, keep_traj=(target == "path1" and split == "test"))
    save_result(r, pol.name, "none", 0, controller=kind, speed=speed)
    return r


def select_speed(rows):
    full = [s for s, r in rows.items() if r["n_complete"] == 20]
    if full:
        return max(full), False
    best = max(r["n_complete"] for r in rows.values())
    return max(s for s, r in rows.items() if r["n_complete"] == best), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["path1", "path2", "path3"])
    a = ap.parse_args()
    sweep = {}
    for kind in SWEPT:
        rows = {s: run(kind, s, "path1", "val")["summary"] for s in SPEEDS}
        sel, fallback = select_speed(rows)
        sweep[kind] = dict(val_path1=rows, selected_speed=sel, fallback_used=fallback)
        print(kind, {s: (r["n_complete"], round(r["steps_mean"])) for s, r in rows.items()}, "->", sel,
              "(fallback)" if fallback else "", flush=True)
    json.dump(sweep, open(os.path.join(config.RESULTS_DIR, "sweep.json"), "w"), indent=1)
    for t in a.targets:
        for k in ("random", "zero", "raster"):
            run(k, 0, t, "test")
        for kind in SWEPT:
            for s in sorted({sweep[kind]["selected_speed"], VMAX}):
                r = run(kind, s, t, "test")
                print(t, kind, s, r["summary"], flush=True)


if __name__ == "__main__":
    main()
