"""
Metric sensitivity: recompute coverage on SAVED path1 test trajectories for rho in RHOS (px).
Coverage_rho = fraction of contour pixels within rho of the trajectory polyline (exact point-to-segment distance,
same swept-capsule definition as the env; rho=8 reproduces the env number up to the 0.1 px trajectory rounding,
checked below).  Reports mean coverage per method and the method ranking at each rho.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import *
from evaluation import make_env

RHOS = [2, 4, 6, 8, 10, 12, 16]


def min_dist(pix, traj):
    """min over segments of distance from each contour pixel to the polyline."""
    P = np.asarray(traj, float).reshape(-1, 2)
    a, b = P[:-1], P[1:]
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(1), 1e-12)
    best = np.full(len(pix), 1e18)
    for s in range(0, len(a), 256):
        A, AB, L = a[s:s + 256], ab[s:s + 256], L2[s:s + 256]
        t = np.clip(((pix[:, None, :] - A[None]) * AB[None]).sum(2) / L[None], 0, 1)
        d = np.linalg.norm(pix[:, None, :] - (A[None] + t[..., None] * AB[None]), axis=2)
        best = np.minimum(best, d.min(1))
    return best


def main(target="path1"):
    pix = make_env(target)._contour_pixels.astype(float)
    # one entry per (method key); learned methods averaged over seeds
    groups = defaultdict(list)
    for r in load(target=target, split="test"):
        if r["method"].startswith("ppo_") and "ckpt" in r["method"]: continue
        if "traj" not in r["episodes"][0]: continue
        key = r["method"] if r["tier"] == "none" else f"{r['method']}|{r['tier']}"
        groups[key].append(r)
    res, check = {}, []
    for key, rs in groups.items():
        per_seed = []
        for r in rs:
            cov = []
            for e in r["episodes"]:
                d = min_dist(pix, e["traj"])
                cov.append([100.0 * (d <= rho).mean() for rho in RHOS])
                check.append(abs(cov[-1][RHOS.index(8)] - e["coverage_pct"]))
            per_seed.append(np.mean(cov, 0))
        res[key] = np.mean(per_seed, 0)
    print(f"rho=8 recomputation vs env coverage: max abs diff {max(check):.3f} pp over {len(check)} episodes")
    order = {i: sorted(res, key=lambda k: -res[k][i]) for i in range(len(RHOS))}
    base = order[RHOS.index(8)]
    L = [r"\begin{tabular}{l" + "r" * len(RHOS) + "}", r"\toprule",
         "Method & " + " & ".join(f"$\\rho$={r}" for r in RHOS) + r" \\", r"\midrule"]
    for k in base:
        L.append(k.replace("_", r"\_").replace("|", " ") + " & " + " & ".join(
            f"{res[k][i]:.1f} ({order[i].index(k) + 1})" for i in range(len(RHOS))) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("metric_sensitivity.tex", "\n".join(L))
    json.dump(dict(rhos=RHOS, coverage={k: list(map(float, v)) for k, v in res.items()},
                   max_abs_diff_rho8=float(max(check))),
              open(os.path.join(ROOT, "results", "metric_sensitivity.json"), "w"), indent=1)
    print("wrote metric_sensitivity.tex/json (cell = mean coverage % (rank))")


if __name__ == "__main__":
    main()
