"""Fair-speed comparison on path1 test: PPO vs greedy / pure pursuit at their validation-selected speed
(results/sweep.json) and at v_max = 15."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import *


def main(target="path1"):
    sweep = json.load(open(os.path.join(ROOT, "results", "sweep.json")))
    L = [r"\begin{tabular}{lrrrrr}", r"\toprule", r"Method & Speed & Coverage (\%) & Complete & Steps & Mean speed \\",
         r"\midrule"]
    for sel in ("ppo_final", "ppo_valsel"):
        rs = load(sel, "full", target)
        if rs:
            L.append(f"PPO full ({sel[4:]}) & -- & {fmt(agg(rs, 'coverage_mean'))} & "
                     f"{sum(r['summary']['n_complete'] for r in rs)}/{20 * len(rs)} & "
                     f"{agg(rs, 'steps_mean')['mean']:.0f} & {agg(rs, 'speed_mean')['mean']:.2f} \\\\")
    L.append(r"\midrule")
    for kind, d in sweep.items():
        for label, s in (("selected", d["selected_speed"]), ("$v_{max}$", 15)):
            r = load(f"{kind}_v{s}", "none", target)
            if r:
                S = r[0]["summary"]
                L.append(f"{kind.replace('_', ' ')} ({label}) & {s} & {S['coverage_mean']:.1f} & {S['n_complete']}/20 & "
                         f"{S['steps_mean']:.0f} & {S['speed_mean']:.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    print("wrote", write(f"fair_speed_{target}.tex", "\n".join(L)))
    # sweep table (validation, path1)
    L = [r"\begin{tabular}{llrrr}", r"\toprule", r"Controller & Speed & Complete (val) & Mean steps & Coverage \\", r"\midrule"]
    for kind, d in sweep.items():
        for s, r in d["val_path1"].items():
            star = "$^*$" if int(s) == d["selected_speed"] else ""
            L.append(f"{kind.replace('_', ' ')} & {s}{star} & {r['n_complete']}/20 & {r['steps_mean']:.0f} & {r['coverage_mean']:.1f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    print("wrote", write("speed_sweep.tex", "\n".join(L)))


if __name__ == "__main__":
    main()
