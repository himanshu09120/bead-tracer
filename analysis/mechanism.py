"""
Mechanism analysis on path1 test episodes of learned policies (pooled over seeds, per tier and selection rule).
 * speed distribution (histogram of per-step speed, world units/s) and fraction of steps above 90% of v_max
 * fraction of all MISSED pixels that belong to episodes ended by the stagnation limit (vs max_steps / complete)
 * miss rate by SIGNED curvature class (concave / straight / convex) with bootstrap CIs over episodes

Curvature: each contour polyline (same Canny/findContours as BeadEnv) is Gaussian-smoothed (sigma = 5 vertices),
kappa = (x'y'' - y'x'') / (x'^2 + y'^2)^1.5, multiplied by sign(signed area) of that loop, so kappa > 0 means the
contour bends TOWARD the loop's own interior (convex) and kappa < 0 away from it (concave).  Classes are by sign:
|kappa| <= KAPPA_STRAIGHT (1/400 px^-1) is 'straight', above is convex, below is concave -- not by tightness.

Writes results/mechanism.json and results/tables/mechanism.tex.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from analysis.common import *
from evaluation import make_env

VMAX = 15.0
KAPPA_STRAIGHT = 1 / 400.0
NAMES = ["concave", "straight", "convex"]


def smooth(x, s=5):
    k = np.arange(-4 * s, 4 * s + 1)
    g = np.exp(-0.5 * (k / s) ** 2)
    g /= g.sum()
    xp = np.concatenate([x[-4 * s:], x, x[:4 * s]])
    return np.convolve(xp, g, mode="same")[4 * s:-4 * s]


def pixel_curvature_class(env):
    """class per contour-pixel-table index: 0 concave, 1 straight, 2 convex, -1 unclassified."""
    gray = cv2.cvtColor(env.target, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    S = env.IMG_SIZE
    table = -np.ones((S, S), dtype=np.int64)
    table[env._contour_pixels[:, 1], env._contour_pixels[:, 0]] = np.arange(len(env._contour_pixels))
    cls = -np.ones(len(env._contour_pixels), dtype=int)
    for c in cnts:
        if cv2.arcLength(c, True) <= 30 or len(c) < 40:
            continue
        P = c[:, 0, :].astype(float)
        x, y = smooth(P[:, 0]), smooth(P[:, 1])
        dx, dy = np.gradient(x), np.gradient(y)
        ddx, ddy = np.gradient(dx), np.gradient(dy)
        k = (dx * ddy - dy * ddx) / np.maximum((dx ** 2 + dy ** 2) ** 1.5, 1e-9)
        area = 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
        k = k * np.sign(area)
        cc = np.where(k > KAPPA_STRAIGHT, 2, np.where(k < -KAPPA_STRAIGHT, 0, 1))
        ix = table[P[:, 1].astype(int), P[:, 0].astype(int)]
        cls[ix[ix >= 0]] = cc[ix >= 0]
    return cls


def boot(vals, n=2000, seed=0):
    vals = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    m = vals[rng.integers(0, len(vals), (n, len(vals)))].mean(1)
    return float(vals.mean()), *map(float, np.percentile(m, [2.5, 97.5]))


def main(target="path1"):
    env = make_env(target)
    cls = pixel_curvature_class(env)
    N = len(cls)
    out = {"class_fraction_of_pixels": {NAMES[i]: float((cls == i).mean()) for i in range(3)},
           "unclassified_fraction": float((cls < 0).mean())}
    L = [r"\begin{tabular}{llrrrrr}", r"\toprule",
         r"Model & Tier & Steps $>0.9v_{max}$ & Missed px from stagnation & Concave miss & Straight miss & Convex miss \\",
         r"\midrule"]
    for sel in ("ppo_final", "ppo_valsel"):
        for tier in TIER_ORDER:
            rs = load(sel, tier, target)
            if not rs:
                continue
            eps = [e for r in rs for e in r["episodes"]]
            speeds = np.concatenate([e["speeds"] for e in eps])
            miss = np.array([len(e["missed_idx"]) for e in eps], float)
            reason = np.array([e["end_reason"] for e in eps])
            miss_mat = np.zeros((len(eps), N), bool)
            for i, e in enumerate(eps):
                miss_mat[i, e["missed_idx"]] = True
            cur = {}
            for ci, nm in enumerate(NAMES):
                px = cls == ci
                cur[nm] = boot(miss_mat[:, px].mean(1) * 100) if px.any() else (float("nan"),) * 3
            hist, edges = np.histogram(speeds, bins=np.linspace(0, VMAX, 31))
            tot = max(miss.sum(), 1)
            o = dict(n_episodes=len(eps), n_seeds=len(rs), speed_hist=hist.tolist(), speed_edges=edges.tolist(),
                     frac_steps_above_0p9_vmax=float((speeds > 0.9 * VMAX).mean()), mean_speed=float(speeds.mean()),
                     missed_from_stagnation=float(miss[reason == "stagnation"].sum() / tot),
                     missed_from_max_steps=float(miss[reason == "max_steps"].sum() / tot),
                     missed_from_complete=float(miss[reason == "complete"].sum() / tot),
                     miss_rate_pct_by_curvature={k: dict(mean=v[0], lo=v[1], hi=v[2]) for k, v in cur.items()})
            out[f"{sel}|{tier}"] = o
            L.append(f"{sel[4:]} & {tier.replace('_', ' ')} & {100 * o['frac_steps_above_0p9_vmax']:.1f}\\% & "
                     f"{100 * o['missed_from_stagnation']:.1f}\\% & "
                     + " & ".join(f"{cur[n][0]:.1f} [{cur[n][1]:.1f}, {cur[n][2]:.1f}]" for n in NAMES) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("mechanism.tex", "\n".join(L))
    json.dump(out, open(os.path.join(ROOT, "results", "mechanism.json"), "w"), indent=1)
    print("wrote mechanism.json / mechanism.tex")


if __name__ == "__main__":
    main()
