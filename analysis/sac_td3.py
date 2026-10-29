"""
SAC / TD3 rows for Tables VII, VIII and X.  Reads only results/raw/{sac,td3}_final__full__s<seed>__<target>__test.json
(written by evaluate_model.py --algo ... --method <algo>_final).

Conventions follow the existing PPO rows:
  per-seed row  : coverage mean +- SD over the 20 starts, completions /20, mean return, mean steps, mean speed
  aggregate row : mean +- sample SD (ddof=1) of the three seed means; completions summed over seeds (/60);
                  return, steps, speed = mean of seed means
  held-out      : per seed, mean over the 10 shapes of per-shape mean coverage +- sample SD over shapes (ddof=1);
                  aggregate = mean +- sample SD of the three per-seed held-out means
Writes results/sac_td3.json and results/tables/sac_td3_{vii,viii,x}.tex
    python -m analysis.sac_td3
"""
import json
import numpy as np
from analysis.common import *

ALGOS = ["sac", "td3"]
SEEDS = [42, 1, 2]
TARGETS = ["path1", "path2", "path3"] + HELDOUT


def get(algo, seed, target):
    return json.load(open(os.path.join(RAW, f"{algo}_final__full__s{seed}__{target}__test.json")))


def seed_row(algo, seed):
    s = get(algo, seed, "path1")["summary"]
    shapes = [get(algo, seed, t)["summary"]["coverage_mean"] for t in HELDOUT]
    return dict(seed=seed, path1=s, path2=get(algo, seed, "path2")["summary"]["coverage_mean"],
                path3=get(algo, seed, "path3")["summary"]["coverage_mean"], heldout_shape_means=shapes,
                heldout_mean=float(np.mean(shapes)), heldout_sd=float(np.std(shapes, ddof=1)),
                end_reasons={k: sum(e["end_reason"] == k for e in get(algo, seed, "path1")["episodes"])
                             for k in ("complete", "max_steps", "stagnation")})


def sd(x):
    return float(np.std(x, ddof=1))


def main():
    res = {}
    for a in ALGOS:
        rows = [seed_row(a, s) for s in SEEDS]
        cov = [r["path1"]["coverage_mean"] for r in rows]
        ho = [r["heldout_mean"] for r in rows]
        res[a] = dict(
            seeds=rows,
            aggregate=dict(
                cov_mean=float(np.mean(cov)), cov_sd_of_seed_means=sd(cov), cov_min=min(cov), cov_max=max(cov),
                n_complete=sum(r["path1"]["n_complete"] for r in rows), n_starts=20 * len(rows),
                ret_mean=float(np.mean([r["path1"]["ret_mean"] for r in rows])),
                steps_mean=float(np.mean([r["path1"]["steps_mean"] for r in rows])),
                speed_mean=float(np.mean([r["path1"]["speed_mean"] for r in rows])),
                path2_mean=float(np.mean([r["path2"] for r in rows])), path2_sd=sd([r["path2"] for r in rows]),
                path3_mean=float(np.mean([r["path3"] for r in rows])), path3_sd=sd([r["path3"] for r in rows]),
                heldout_mean=float(np.mean(ho)), heldout_sd_of_seed_means=sd(ho)))
    json.dump(res, open(os.path.join(ROOT, "results", "sac_td3.json"), "w"), indent=1)

    N = {"sac": "SAC", "td3": "TD3"}
    vii, viii, x = [], [], []
    for a in ALGOS:
        for r in res[a]["seeds"]:
            p = r["path1"]
            vii.append(f"{N[a]}, seed {r['seed']}, $\\phi = 0.95$ & {p['coverage_mean']:.1f} $\\pm$ {p['coverage_sd']:.1f} & "
                       f"{p['n_complete']}/20 & {p['ret_mean']:.1f} & {p['steps_mean']:.0f} \\\\")
            x.append(f"{N[a]} trained on path1 (seed {r['seed']}) & {p['coverage_mean']:.1f} & {r['path2']:.1f} & "
                     f"{r['path3']:.1f} & {r['heldout_mean']:.1f} $\\pm$ {r['heldout_sd']:.1f} \\\\")
        g = res[a]["aggregate"]
        vii.append(f"{N[a]}, three seeds, $\\phi = 0.95$ (SD of the three seed means) & {g['cov_mean']:.1f} $\\pm$ "
                   f"{g['cov_sd_of_seed_means']:.1f} & {g['n_complete']}/{g['n_starts']} & {g['ret_mean']:.1f} & {g['steps_mean']:.0f} \\\\")
        viii.append(f"{N[a]} (final, 3 seeds) & {g['cov_mean']:.1f} & {g['n_complete']}/{g['n_starts']} & "
                    f"{g['steps_mean']:.0f} & {g['speed_mean']:.2f} \\\\")
        x.append(f"{N[a]} trained on path1 (mean of 3 seeds, $\\pm$ SD of seed means) & {g['cov_mean']:.1f} $\\pm$ "
                 f"{g['cov_sd_of_seed_means']:.1f} & {g['path2_mean']:.1f} $\\pm$ {g['path2_sd']:.1f} & "
                 f"{g['path3_mean']:.1f} $\\pm$ {g['path3_sd']:.1f} & {g['heldout_mean']:.1f} $\\pm$ {g['heldout_sd_of_seed_means']:.1f} \\\\")
    for n, L in (("vii", vii), ("viii", viii), ("x", x)):
        print("wrote", write(f"sac_td3_{n}.tex", "\n".join(L)))
        print("\n".join(L), "\n")


if __name__ == "__main__":
    main()
