"""
Slow-tracer test (evaluation only): a scripted pure-pursuit tracer follows the contour at several speeds on
path1 (10 test starts = every other test start) with a 20,000-step cap so slow tracers can finish.
Reports total return, coverage and the reward decomposition per speed:
  coverage reward = _cov_r * 100 * (new px)/N   -- identical for the same pixels regardless of speed
  continuity bonus = 0.15 * continuous_new/new_px on every covering step -- paid per STEP, so it grows when the
                     same pixels are covered in more (slower) steps.
Run: python -m analysis.slow_tracer      -> results/slow_tracer.json + results/tables/slow_tracer.tex
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from analysis.common import *
from evaluation import make_env, test_starts
from baselines import PurePursuitPolicy

SPEEDS = [2, 3, 4, 6, 8, 10, 12]


def run(speed, env, start, i):
    pol = PurePursuitPolicy(speed)
    obs, _ = env.reset(seed=1000 + i, options={"fixed_start": start})
    pol.reset(env, i)
    spawn = env.covered_pixels
    tot = cont = 0.0
    done = False
    while not done:
        obs, r, te, tr, info = env.step(pol.act(obs, env))
        tot += r
        if env._last_new_px:
            cont += env._continuity_r * env._last_continuous_new / env._last_new_px
        done = te or tr
    N = env.total_contour_pixels
    return dict(ret=tot, coverage=info["coverage_pct"], steps=info["steps"], complete=bool(info["task_complete"]),
                cov_reward=env._cov_r * 100.0 * (env.covered_pixels - spawn) / N, continuity=cont,
                completion_bonus=env._completion_r if info["task_complete"] else 0.0)


def main():
    env = make_env("path1", max_steps=20000)
    starts = test_starts(env)[::2]
    out = {}
    for v in SPEEDS:
        rows = [run(v, env, s, 2 * i) for i, s in enumerate(starts)]
        out[v] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        out[v]["n_complete"] = int(sum(r["complete"] for r in rows)); out[v]["n"] = len(rows)
        print(v, out[v], flush=True)
    json.dump(out, open(os.path.join(ROOT, "results", "slow_tracer.json"), "w"), indent=1)
    L = [r"\begin{tabular}{rrrrrrr}", r"\toprule",
         r"Speed & Coverage (\%) & Complete & Steps & Return & Coverage rew. & Continuity \\", r"\midrule"]
    for v, o in out.items():
        L.append(f"{v} & {o['coverage']:.1f} & {o['n_complete']}/{o['n']} & {o['steps']:.0f} & {o['ret']:.1f} & "
                 f"{o['cov_reward']:.1f} & {o['continuity']:.1f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("slow_tracer.tex", "\n".join(L))


if __name__ == "__main__":
    main()
