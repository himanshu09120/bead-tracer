"""
Evaluate one training run (final model and validation-selected checkpoint) on test starts of every target.
Model selection uses path1 VALIDATION starts only (earliest checkpoint wins ties).

    python evaluate_run.py --run no_direction_s1 --tier no_direction
    python evaluate_run.py --run full_s42 --tier full --ckpt-test      # also test coverage of every checkpoint
Results: results/raw/ppo_final__<tier>__s<seed>__<target>__test.json, ppo_valsel__...; results/raw/selection_*.json
Skips results that already exist, so it is resumable.
"""
import argparse, json, os, re
import config
from evaluation import evaluate, save_result, result_path, PPOPolicy
from evaluation.runs import run_files, best_checkpoint, checkpoints

TARGETS = ["path1", "path2", "path3"] + [f"heldout{i:02d}" for i in range(10)]


def ev(method, path, tier, seed, target, split, phi, **meta):
    p = result_path(method, tier, seed, target, split)
    if os.path.exists(p):
        return
    r = evaluate(PPOPolicy(path), target, split, tier, phi, keep_traj=(target == "path1" and split == "test"))
    save_result(r, method, tier, seed, model=os.path.relpath(path, config.PROJECT_ROOT), **meta)
    print(method, target, split, r["summary"]["coverage_mean"], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tier", required=True)
    ap.add_argument("--phi", type=float, default=0.95)
    ap.add_argument("--targets", nargs="+", default=TARGETS)
    ap.add_argument("--ckpt-test", action="store_true")
    a = ap.parse_args()
    seed = int(re.search(r"_s(\d+)", a.run).group(1))
    final, _ = run_files(a.run)
    k, ckpt, scores = best_checkpoint(a.run, a.tier, "path1", a.phi)
    json.dump(dict(run=a.run, selected_checkpoint_steps=k, val_coverage_by_checkpoint=scores,
                   rule="argmax mean coverage on path1 validation starts, earliest on ties"),
              open(os.path.join(config.RESULTS_DIR, "raw", f"selection__{a.tier}__s{seed}.json"), "w"), indent=1)
    print("selected checkpoint", k, flush=True)
    for t in a.targets:
        ev("ppo_final", final, a.tier, seed, t, "test", a.phi)
        ev("ppo_valsel", ckpt, a.tier, seed, t, "test", a.phi, selected_checkpoint_steps=k)
    if a.ckpt_test:
        for kk, f in checkpoints(a.run):
            ev(f"ppo_{a.run}_ckpt{kk}", f, a.tier, seed, "path1", "test", a.phi, checkpoint_steps=kk)


if __name__ == "__main__":
    main()
