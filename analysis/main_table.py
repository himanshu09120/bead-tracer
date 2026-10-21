"""Main table: every method x tier.  mean +/- sample SD over seeds [95% bootstrap CI of seed means].
Learned methods: one value per seed (each = mean over 20 test starts).  Controllers: single deterministic run
(n=1, no SD/CI).  Completion = completed episodes summed over seeds / (20*seeds)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import *


def rows_for(target):
    rows = []
    for sel, label in (("ppo_final", "final"), ("ppo_valsel", "val-selected")):
        for tier in TIER_ORDER:
            rs = load(sel, tier, target)
            if rs:
                rows.append((f"PPO {tier} ({label})", rs))
    names = sorted({os.path.basename(f).split("__")[0] for f in glob.glob(os.path.join(RAW, "*__none__*"))})
    for n in names:
        rs = load(n, "none", target)
        if rs:
            rows.append((n.replace("_", r"\_"), rs))
    return rows


def table(target):
    L = [r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Method & Seeds & Coverage (\%) & Complete & Return & Steps \\", r"\midrule"]
    for name, rs in rows_for(target):
        comp = sum(r["summary"]["n_complete"] for r in rs)
        tot = sum(r["n_starts"] for r in rs)
        L.append(f"{name} & {len(rs)} & {fmt(agg(rs, 'coverage_mean'))} & {comp}/{tot} & "
                 f"{agg(rs, 'ret_mean')['mean']:.1f} & {agg(rs, 'steps_mean')['mean']:.0f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def main():
    for t in ["path1", "path2", "path3"]:
        print("wrote", write(f"main_{t}.tex", table(t)))
    # held-out procedural shapes: per seed, average over the 10 shapes first
    L = [r"\begin{tabular}{lrrr}", r"\toprule", r"Method & Seeds & Coverage (\%) & Complete \\", r"\midrule"]
    for sel, label in (("ppo_final", "final"), ("ppo_valsel", "val-selected")):
        for tier in TIER_ORDER:
            per_seed = defaultdict(list)
            for t in HELDOUT:
                for r in load(sel, tier, t):
                    per_seed[r["seed"]].append(r["summary"])
            if per_seed:
                cov = [np.mean([s["coverage_mean"] for s in v]) for v in per_seed.values()]
                comp = sum(sum(s["n_complete"] for s in v) for v in per_seed.values())
                lo, hi = boot_ci(cov)
                sd = np.std(cov, ddof=1) if len(cov) > 1 else float("nan")
                L.append(f"PPO {tier} ({label}) & {len(cov)} & {np.mean(cov):.1f} $\pm$ {sd:.1f} [{lo:.1f}, {hi:.1f}] "
                         f"& {comp}/{len(per_seed) * 200} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    print("wrote", write("main_heldout.tex", "\n".join(L)))


if __name__ == "__main__":
    main()
