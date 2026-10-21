"""Figures: trajectory coloured by speed (best & worst episode), miss-rate map, checkpoint curves (all seeds).
Reads results/raw only.  Output: results/figures/*.pdf and .png"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from analysis.common import *
from evaluation import make_env


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def traj_fig(target="path1", sel="ppo_final", tier="full"):
    env = make_env(target)
    eps = [(r["seed"], e) for r in load(sel, tier, target) for e in r["episodes"]]
    if not eps:
        return
    best = max(eps, key=lambda x: x[1]["coverage_pct"])
    worst = min(eps, key=lambda x: x[1]["coverage_pct"])
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    for ax, (sd, e), ttl in zip(axs, (best, worst), ("best", "worst")):
        cp = env._contour_pixels
        ax.scatter(cp[:, 0], cp[:, 1], s=1, c="0.75")
        P = np.array(e["traj"]).reshape(-1, 2)
        seg = np.stack([P[:-1], P[1:]], 1)
        lc = LineCollection(seg, cmap="viridis", norm=plt.Normalize(0, 15), linewidths=1.2)
        lc.set_array(np.array(e["speeds"])[: len(seg)])
        ax.add_collection(lc)
        ax.plot(*P[0], "o", c="tab:red", ms=5)
        ax.set_xlim(0, 800); ax.set_ylim(800, 0); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{ttl}: seed {sd}, {e['coverage_pct']:.1f}% cov, {e['steps']} steps")
    fig.colorbar(lc, ax=axs, shrink=0.7, label="speed (units/s)")
    save(fig, f"trajectory_{sel}_{tier}")


def miss_map(target="path1", sel="ppo_final", tier="full"):
    env = make_env(target)
    eps = [e for r in load(sel, tier, target) for e in r["episodes"]]
    if not eps:
        return
    m = np.zeros(env.total_contour_pixels)
    for e in eps:
        m[e["missed_idx"]] += 1
    m /= len(eps)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    cp = env._contour_pixels
    sc = ax.scatter(cp[:, 0], cp[:, 1], c=m, s=4, cmap="magma_r", vmin=0, vmax=1)
    ax.set_xlim(0, 800); ax.set_ylim(800, 0); ax.set_aspect("equal"); ax.axis("off")
    fig.colorbar(sc, shrink=0.7, label=f"miss rate over {len(eps)} episodes")
    save(fig, f"missmap_{sel}_{tier}")


def ckpt_curves():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    colors = dict(zip(TIER_ORDER, plt.cm.tab10.colors))
    for f in sorted(glob.glob(os.path.join(RAW, "selection__*__s*.json"))):
        d = json.load(open(f))
        _, tier, s = os.path.basename(f)[:-5].split("__")
        v = d["val_coverage_by_checkpoint"]
        ks = sorted(int(k) for k in v)
        axs[0].plot(np.array(ks) / 1e3, [v[str(k)] for k in ks], c=colors.get(tier, "k"), alpha=0.7,
                    label=f"{tier} {s}")
    axs[0].set(title="validation coverage per checkpoint (path1)", xlabel="steps (k)", ylabel="coverage (%)")
    axs[0].legend(fontsize=6, ncol=2)
    byrun = defaultdict(dict)
    for r in load(prefix="ppo_", target="path1", split="test"):
        m = re.match(r"ppo_(.+)_ckpt(\d+)$", r["method"])
        if m:
            byrun[m.group(1)][int(m.group(2))] = r["summary"]["coverage_mean"]
    for run, v in byrun.items():
        ks = sorted(v)
        axs[1].plot(np.array(ks) / 1e3, [v[k] for k in ks], marker="o", ms=3, label=run)
    axs[1].set(title="test coverage per checkpoint (path1)", xlabel="steps (k)")
    if byrun:
        axs[1].legend(fontsize=6)
    save(fig, "checkpoint_curves")


def main():
    traj_fig()
    miss_map()
    ckpt_curves()
    print("figures written to results/figures")


if __name__ == "__main__":
    main()
