"""
evaluate.py -- Deterministic evaluation of a trained PPO model on BeadEnv.

Separate from training by design. Uses the environment's OWN deterministic
evaluation support (`env.eval_start_points(n)` + `env.reset(options={
"fixed_start": ...})`) so every evaluation run replays the same n starting
states -- no random resets, no competing coverage metric. All reported
numbers (coverage, coverage_pct, covered_pixels, total_contour_pixels) are
read directly from the environment's own `coverage_info()` / properties.

Usage:
    python evaluate.py --model models/bead_ppo_final.zip
    python evaluate.py --model models/bead_ppo_final.zip --image target_images/path1.png --n-points 30
"""

import argparse
import json
import os
import time

import numpy as np
from stable_baselines3 import PPO

import config
from env import BeadEnv
from train import load_target_image


# =============================================================================
# Episode rollout
# =============================================================================

def run_episode(env: BeadEnv, model: PPO, start_point, deterministic: bool = True,
                 max_steps: int = None):
    """Runs one full episode from a fixed (px, py) contour starting point,
    using model.predict(..., deterministic=deterministic) at every step.
    Returns a dict of metrics read straight from the environment -- the
    environment's coverage bookkeeping is the only source of truth here.
    """
    obs, info = env.reset(seed=None, options={"fixed_start": start_point})
    total_reward = 0.0
    terminated = truncated = False
    steps = 0
    cap = max_steps if max_steps is not None else env.max_steps

    while not (terminated or truncated) and steps < cap:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1

    return {
        "start_point": [int(start_point[0]), int(start_point[1])],
        "coverage": info["coverage"],
        "coverage_pct": info["coverage_pct"],
        "covered_pixels": info["covered_pixels"],
        "total_contour_pixels": info["total_contour_pixels"],
        "reward": total_reward,
        "steps": info["steps"],
        "task_complete": bool(info["task_complete"]),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


# =============================================================================
# Full evaluation sweep
# =============================================================================

def evaluate_model(model_path: str, image_path: str, n_points: int, seed: int,
                    deterministic: bool = True, max_steps: int = None,
                    render_dir: str = None, render_every: int = 1):
    """Evaluates a trained model over n_points deterministic, evenly spaced
    starting points on the target contour (env.eval_start_points(n_points)).
    Optionally renders env.render_coverage() for a subset of episodes.
    Returns (per_episode_results, summary_dict).
    """
    img = load_target_image(image_path)
    env = BeadEnv(target_image=img, curriculum=False, **config.ENV_KWARGS)
    env.action_space.seed(seed)

    model = PPO.load(model_path, device="auto")

    start_points = env.eval_start_points(n_points)
    print(f"Evaluating {model_path}")
    print(f"  image           : {image_path}")
    print(f"  contour pixels  : {env.total_contour_pixels}")
    print(f"  start points    : {n_points} (deterministic, evenly spaced)")
    print(f"  deterministic   : {deterministic}")

    results = []
    for i, start_point in enumerate(start_points):
        r = run_episode(env, model, start_point, deterministic=deterministic,
                         max_steps=max_steps)
        results.append(r)
        print(f"  [{i + 1:>3}/{n_points}] start={tuple(r['start_point'])!s:<16} "
              f"coverage={r['coverage_pct']:6.2f}%  "
              f"covered={r['covered_pixels']:>6}/{r['total_contour_pixels']:<6}  "
              f"reward={r['reward']:+8.2f}  steps={r['steps']:>5}  "
              f"complete={r['task_complete']}")

        if render_dir is not None and (i % render_every == 0):
            os.makedirs(render_dir, exist_ok=True)
            out_path = os.path.join(render_dir, f"eval_point_{i:02d}.png")
            env.render_coverage(
                save_path=out_path,
                title=f"Eval start #{i} -- coverage {r['coverage_pct']:.2f}%",
            )

    env.close()

    coverages = np.array([r["coverage_pct"] for r in results], dtype=np.float64)
    completions = np.array([r["task_complete"] for r in results], dtype=np.float64)
    rewards = np.array([r["reward"] for r in results], dtype=np.float64)
    steps_arr = np.array([r["steps"] for r in results], dtype=np.float64)

    summary = {
        "model_path": model_path,
        "image_path": image_path,
        "n_points": n_points,
        "deterministic": deterministic,
        "total_contour_pixels": int(env.total_contour_pixels),
        "mean_coverage_pct": float(coverages.mean()),
        "std_coverage_pct": float(coverages.std()),
        "min_coverage_pct": float(coverages.min()),
        "max_coverage_pct": float(coverages.max()),
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "mean_steps": float(steps_arr.mean()),
        "completion_rate": float(completions.mean()),
        "n_completed": int(completions.sum()),
    }
    return results, summary


def print_summary(summary: dict):
    print("\n" + "=" * 78)
    print("  EVALUATION SUMMARY")
    print("=" * 78)
    print(f"  model               : {summary['model_path']}")
    print(f"  image               : {summary['image_path']}")
    print(f"  total contour pixels: {summary['total_contour_pixels']}")
    print(f"  start points        : {summary['n_points']}")
    print(f"  deterministic       : {summary['deterministic']}")
    print("-" * 78)
    print(f"  mean coverage       : {summary['mean_coverage_pct']:.2f}%")
    print(f"  std  coverage       : {summary['std_coverage_pct']:.2f} pp")
    print(f"  min  coverage       : {summary['min_coverage_pct']:.2f}%")
    print(f"  max  coverage       : {summary['max_coverage_pct']:.2f}%")
    print(f"  mean reward         : {summary['mean_reward']:+.2f}")
    print(f"  mean steps          : {summary['mean_steps']:.1f}")
    print(f"  completion rate     : {summary['completion_rate'] * 100:.1f}% "
          f"({summary['n_completed']}/{summary['n_points']})")
    print("=" * 78)


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained PPO model on BeadEnv.")
    p.add_argument("--model", type=str, default=config.FINAL_MODEL_PATH,
                    help="Path to a trained model .zip.")
    p.add_argument("--image", type=str, default=config.DEFAULT_IMAGE_PATH,
                    help="Path to the target image.")
    p.add_argument("--n-points", type=int, default=config.EVAL_N_START_POINTS,
                    help="Number of deterministic evaluation starting points.")
    p.add_argument("--seed", type=int, default=config.SEED,
                    help="Random seed (used only to seed the action space; "
                         "starting points and rollout are deterministic).")
    p.add_argument("--stochastic", action="store_true",
                    help="Use stochastic (sampled) actions instead of the "
                         "deterministic policy mean.")
    p.add_argument("--max-steps", type=int, default=config.EVAL_MAX_STEPS,
                    help="Per-episode step cap during evaluation.")
    p.add_argument("--render-every", type=int, default=1,
                    help="Render every Nth evaluation episode to results/ (1 = all).")
    p.add_argument("--no-render", action="store_true",
                    help="Skip saving coverage visualizations to results/.")
    p.add_argument("--tag", type=str, default=None,
                    help="Extra tag appended to output filenames (default: timestamp).")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed_for_reporting = args.seed  # evaluation is deterministic by construction

    tag = args.tag or time.strftime("%Y%m%d_%H%M%S")
    render_dir = None if args.no_render else os.path.join(config.RESULTS_DIR, f"eval_{tag}")

    results, summary = evaluate_model(
        model_path=args.model,
        image_path=args.image,
        n_points=args.n_points,
        seed=set_seed_for_reporting,
        deterministic=not args.stochastic,
        max_steps=args.max_steps,
        render_dir=render_dir,
        render_every=max(args.render_every, 1),
    )

    print_summary(summary)

    out_path = os.path.join(config.EVALUATIONS_DIR, f"eval_{tag}.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "episodes": results}, f, indent=2)
    print(f"\nSaved full evaluation results to {out_path}")
    if render_dir is not None:
        print(f"Saved coverage visualizations to {render_dir}/")


if __name__ == "__main__":
    main()
