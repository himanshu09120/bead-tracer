"""
benchmark/evaluate_ours.py

Runs OUR OWN trained PPO model through the exact same evaluation protocol as
the project's own evaluate.py (same BeadEnv construction, same
eval_start_points(), same deterministic model.predict() loop) but ALSO
computes path_length and redundancy_rate via benchmark/common/metrics.py --
the two extra metrics evaluate.py doesn't report, needed to compare against
every baseline in this benchmark on the same footing.

Does not modify env.py, config.py, train.py, or evaluate.py, and does not
change how the model was trained or evaluated -- it only adds the two extra
metric computations on top of the same rollout.

Usage:
    python benchmark/evaluate_ours.py --model models/bead_ppo_final.zip \
        --image target_images/path1.png
"""

import argparse
import os
import sys

from stable_baselines3 import PPO

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

import config  # noqa: E402
from env import BeadEnv  # noqa: E402
from train import load_target_image  # noqa: E402
from common.metrics import (  # noqa: E402
    EpisodeResult, aggregate_results, path_length_from_positions,
    redundancy_rate_from_productive_steps, save_results,
)


def run_episode(env: BeadEnv, model: PPO, start_point, episode_index: int,
                 deterministic: bool = True, max_steps: int = None):
    obs, info = env.reset(seed=None, options={"fixed_start": start_point})
    cap = max_steps if max_steps is not None else env.max_steps

    positions = [env.get_pixel_pos_f()]
    productive_flags = []
    prev_covered = info["covered_pixels"]
    total_reward = 0.0

    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < cap:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        positions.append(env.get_pixel_pos_f())
        productive_flags.append(info["covered_pixels"] > prev_covered)
        prev_covered = info["covered_pixels"]

    return EpisodeResult(
        method="ppo_ours",
        episode_index=episode_index,
        seed=None,
        start_point=[int(start_point[0]), int(start_point[1])],
        coverage_pct=info["coverage_pct"],
        success=bool(info["task_complete"]),
        path_length=path_length_from_positions(positions),
        steps=info["steps"],
        redundancy_rate=redundancy_rate_from_productive_steps(productive_flags),
        reward=total_reward,
        extra={
            "covered_pixels": info["covered_pixels"],
            "total_contour_pixels": info["total_contour_pixels"],
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        },
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=config.FINAL_MODEL_PATH)
    p.add_argument("--image", type=str, default=config.DEFAULT_IMAGE_PATH)
    p.add_argument("--n-points", type=int, default=config.EVAL_N_START_POINTS)
    p.add_argument("--max-steps", type=int, default=config.EVAL_MAX_STEPS)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--out-dir", type=str,
                    default=os.path.join(_PROJECT_ROOT, "benchmark", "results"))
    args = p.parse_args()

    img = load_target_image(args.image)
    env = BeadEnv(target_image=img, curriculum=False, **config.ENV_KWARGS)
    model = PPO.load(args.model, device="auto")

    start_points = env.eval_start_points(args.n_points)
    episodes = []
    for i, sp in enumerate(start_points):
        r = run_episode(env, model, sp, i, deterministic=not args.stochastic,
                         max_steps=args.max_steps)
        episodes.append(r)
        print(f"  [ppo_ours] [{i + 1:>3}/{len(start_points)}] "
              f"start={tuple(r.start_point)!s:<16} coverage={r.coverage_pct:6.2f}%  "
              f"steps={r.steps:>5}  redundancy={r.redundancy_rate:.3f}  "
              f"complete={r.success}")
    env.close()

    summary = aggregate_results(episodes)
    meta = {
        "method": "ppo_ours",
        "model_path": args.model,
        "image": args.image,
        "n_points": args.n_points,
        "deterministic": not args.stochastic,
        "completion_frac": config.ENV_KWARGS["completion_frac"],
        "environment": "BeadEnv (this project's own, unmodified environment)",
        "notes": "Same protocol as evaluate.py (same eval_start_points(), same "
                 "deterministic PPO.predict() loop); path_length and "
                 "redundancy_rate added here via the shared benchmark metrics "
                 "module for cross-method comparison.",
    }
    json_path, csv_path = save_results(args.out_dir, "ppo_ours", episodes, summary, meta)
    print(f"\n[ppo_ours] summary: mean_coverage={summary['mean_coverage_pct']:.2f}% "
          f"success_rate={summary['success_rate']*100:.1f}% "
          f"mean_redundancy={summary['mean_redundancy_rate']:.3f}")
    print(f"[ppo_ours] saved: {json_path}, {csv_path}")


if __name__ == "__main__":
    main()
