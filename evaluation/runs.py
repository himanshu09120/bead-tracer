"""Run registry + model-selection rules (final model / best checkpoint on VALIDATION starts)."""
import glob, os, re
import config
from .core import evaluate, save_result, result_path, PPOPolicy

RUNS = os.path.join(config.PROJECT_ROOT, "runs")


def run_files(name):
    """-> (final_zip, checkpoint_dir).  The pre-existing seed-42 run lives in models/ + checkpoints/ and is
    referenced in place (never copied or modified)."""
    if name == "full_s42":
        return os.path.join(config.MODELS_DIR, "bead_ppo_final.zip"), config.CHECKPOINTS_DIR
    d = os.path.join(RUNS, name)
    return os.path.join(d, "final.zip"), os.path.join(d, "checkpoints")


def checkpoints(name):
    _, ck = run_files(name)
    out = []
    for f in glob.glob(os.path.join(ck, "bead_ppo_*_steps.zip")):
        out.append((int(re.search(r"_(\d+)_steps", f).group(1)), f))
    return sorted(out)


def validation_scores(name, obs_mode, target="path1", phi=0.95):
    """Validation coverage of every checkpoint (cached in results/raw as method ppo_ckpt<k>, split val)."""
    scores = {}
    seed = int(re.search(r"_s(\d+)", name).group(1))
    for k, f in checkpoints(name):
        method = f"ppo_{name}_ckpt{k}"
        p = result_path(method, obs_mode, seed, target, "val")
        if os.path.exists(p):
            import json
            r = json.load(open(p))
        else:
            r = evaluate(PPOPolicy(f), target, "val", obs_mode, phi, keep_traj=False)
            save_result(r, method, obs_mode, seed, model=os.path.relpath(f, config.PROJECT_ROOT))
        scores[k] = r["summary"]["coverage_mean"]
    return scores


def best_checkpoint(name, obs_mode, target="path1", phi=0.95):
    """argmax validation coverage; EARLIEST checkpoint wins ties.  Test starts are never used."""
    s = validation_scores(name, obs_mode, target, phi)
    best = max(s.values())
    k = min(k for k, v in s.items() if v == best)
    return k, dict(checkpoints(name))[k], s
