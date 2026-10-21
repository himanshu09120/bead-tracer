"""
Evaluation harness.  One entry point -- evaluate(policy, target, split, ...) -- for PPO models and
scripted controllers alike.

Start points
------------
test starts : env.eval_start_points(20) = indices round(linspace(0, N-1, 20)) into the raster-scan
              (row-major, y then x) ordered contour-pixel table.  This is the repo's own definition.
val starts  : 20 further pixels.  With test spacing d = (N-1)/19, slots 0..18 sit exactly HALFWAY between
              consecutive test starts, index round(i*d + d/2).  There are only 19 gaps between 20 test starts,
              so slot 19 is the quarter point of the first gap, round(d/4) (wrapping to halfway would land
              next to slot 0 and duplicate it).  Validation starts never coincide with test starts.
              Model selection uses ONLY validation starts.
"""
import json, os, sys
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import config
from env import BeadEnv

RAW_DIR = os.path.join(config.RESULTS_DIR, "raw")


# ---------------------------------------------------------------- targets / env
def target_file(target):
    if target.startswith("path"):
        return os.path.join(config.TARGET_IMAGES_DIR, f"{target}.png")
    if target.startswith("heldout"):
        return os.path.join(ROOT, "shapes", "heldout", f"{target}.png")
    if target.startswith("train"):
        return os.path.join(ROOT, "shapes", "train", f"{target}.png")
    raise ValueError(target)


def load_target(target):
    img = cv2.imread(target_file(target))
    if img is None:
        raise FileNotFoundError(target_file(target))
    return img


def make_env(target, obs_mode="full", phi=0.95, **kw):
    kwargs = dict(config.ENV_KWARGS, completion_frac=phi, obs_mode=obs_mode)
    kwargs.update(kw)
    return BeadEnv(load_target(target), curriculum=False, **kwargs)


# ---------------------------------------------------------------- start points
def test_starts(env, n=20):
    return env.eval_start_points(n)


def val_starts(env, n=20):
    N = len(env._contour_pixels)
    d = (N - 1) / (n - 1)
    idx = [int(round(i * d + d / 2)) for i in range(n - 1)] + [int(round(d / 4))]
    return [(int(env._contour_pixels[i, 0]), int(env._contour_pixels[i, 1])) for i in idx]


# ---------------------------------------------------------------- policies
class PPOPolicy:
    """Deterministic SB3 PPO, always on CPU (CPU and CUDA give different chaotic rollouts)."""
    def __init__(self, path):
        import torch
        from stable_baselines3 import PPO
        torch.set_num_threads(1)
        self.path, self.model = path, PPO.load(path, device="cpu")

    def reset(self, env, i): pass

    def act(self, obs, env):
        return self.model.predict(obs, deterministic=True)[0]


# ---------------------------------------------------------------- rollout
def run_episode(policy, env, start, i, keep_traj=True):
    obs, info = env.reset(seed=1000 + i, options={"fixed_start": start})
    policy.reset(env, i)
    ret, speeds, term, trunc = 0.0, [], False, False
    while not (term or trunc):
        obs, r, term, trunc, info = env.step(policy.act(obs, env))
        ret += float(r)
        speeds.append(float(np.hypot(env.vel_x, env.vel_y)))
    if term:
        reason = "complete"
    elif env.step_count >= env.max_steps:
        reason = "max_steps"
    else:
        reason = "stagnation"
    ep = dict(start=[int(start[0]), int(start[1])], coverage_pct=float(info["coverage_pct"]),
              covered_pixels=int(info["covered_pixels"]), total_pixels=int(info["total_contour_pixels"]),
              complete=bool(info["task_complete"]), ret=ret, steps=int(info["steps"]),
              mean_speed=float(np.mean(speeds)) if speeds else 0.0, end_reason=reason,
              missed_idx=np.flatnonzero(~env._cov_mask).tolist() if keep_traj else None)
    if keep_traj:
        ep["traj"] = np.round(np.asarray(env._traj, dtype=np.float64), 1).ravel().tolist()
        ep["speeds"] = np.round(speeds, 2).tolist()
    return ep


def evaluate(policy, target, split, obs_mode="full", phi=0.95, n=20, keep_traj=True, env=None):
    env = env or make_env(target, obs_mode, phi)
    starts = {"test": test_starts, "val": val_starts}[split](env, n)
    eps = [run_episode(policy, env, s, i, keep_traj) for i, s in enumerate(starts)]
    cov = np.array([e["coverage_pct"] for e in eps])
    return dict(target=target, split=split, obs_mode=obs_mode, phi=phi, n_starts=n,
                n_contour_pixels=int(env.total_contour_pixels), episodes=eps,
                summary=dict(coverage_mean=float(cov.mean()), coverage_sd=float(cov.std()),
                             n_complete=int(sum(e["complete"] for e in eps)),
                             ret_mean=float(np.mean([e["ret"] for e in eps])),
                             steps_mean=float(np.mean([e["steps"] for e in eps])),
                             speed_mean=float(np.mean([e["mean_speed"] for e in eps]))))


# ---------------------------------------------------------------- storage
def result_path(method, tier, seed, target, split):
    return os.path.join(RAW_DIR, f"{method}__{tier}__s{seed}__{target}__{split}.json")


def save_result(res, method, tier, seed, **meta):
    import torch, stable_baselines3
    res = dict(res, method=method, tier=tier, seed=seed, torch=torch.__version__,
               sb3=stable_baselines3.__version__, **meta)
    p = result_path(method, tier, seed, res["target"], res["split"])
    os.makedirs(RAW_DIR, exist_ok=True)
    json.dump(res, open(p, "w"), separators=(",", ":"))
    return p
