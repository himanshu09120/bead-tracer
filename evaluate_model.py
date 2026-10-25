"""
Evaluate one standalone model file (final model, no checkpoint selection) on test starts of the given targets,
with the same harness as evaluate_run.py (CPU, 1 torch thread, deterministic, phi=0.95 evaluation, rho=8 px).
Used for models outside runs/ (e.g. the unrecorded-seed phi=0.80 model) and for the seeded phi=0.80 re-run.

    python evaluate_model.py --model models/bead_ppo_20260819_163258_final.zip --label m163258 --seed unrecorded
Results: results/raw/ppo_<label>__full__s<seed>__<target>__test.json (skips existing files).
"""
import argparse, os
import config
from evaluation import evaluate, save_result, result_path, PPOPolicy

TARGETS = ["path1", "path2", "path3"] + [f"heldout{i:02d}" for i in range(10)]

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--label", required=True)
ap.add_argument("--seed", required=True, help="training seed, or the literal 'unrecorded'")
ap.add_argument("--eval-phi", type=float, default=0.95)
ap.add_argument("--train-phi", type=float, default=None, help="metadata only")
ap.add_argument("--targets", nargs="+", default=TARGETS)
a = ap.parse_args()

pol = None
for t in a.targets:
    method = f"ppo_{a.label}"
    if os.path.exists(result_path(method, "full", a.seed, t, "test")):
        continue
    pol = pol or PPOPolicy(a.model)
    r = evaluate(pol, t, "test", "full", a.eval_phi, keep_traj=(t == "path1"))
    save_result(r, method, "full", a.seed, model=os.path.relpath(a.model, config.PROJECT_ROOT), train_phi=a.train_phi)
    s = r["summary"]
    print(t, f"{s['coverage_mean']:.2f} +- {s['coverage_sd']:.2f}  {s['n_complete']}/20  ret {s['ret_mean']:.1f}  steps {s['steps_mean']:.0f}", flush=True)
