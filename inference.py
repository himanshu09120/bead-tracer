"""
inference.py -- Run a trained PPO model on a target image and report/save
its bead-tracing result.

Usage:
    python inference.py --model models/bead_ppo_final.zip --image target_images/path1.png
    python inference.py --model models/bead_ppo_final.zip --image target_images/path1.png --start 400,100
    python inference.py --model models/bead_ppo_final.zip --image target_images/path1.png --stochastic
"""

import argparse
import os
import time

from stable_baselines3 import PPO

import config
from env import BeadEnv
from train import load_target_image


def parse_start(value: str):
    """Parses '--start x,y' into an (x, y) int tuple, or returns None."""
    if value is None:
        return None
    x_str, y_str = value.split(",")
    return (int(x_str.strip()), int(y_str.strip()))


def run_inference(model_path: str, image_path: str, start=None,
                   deterministic: bool = True, max_steps: int = None,
                   output_path: str = None):
    img = load_target_image(image_path)
    env = BeadEnv(target_image=img, curriculum=(start is None), **config.ENV_KWARGS)

    model = PPO.load(model_path, device="auto")

    options = {"fixed_start": start} if start is not None else None
    obs, info = env.reset(seed=config.SEED, options=options)

    cap = max_steps if max_steps is not None else env.max_steps
    total_reward = 0.0
    terminated = truncated = False
    steps = 0

    while not (terminated or truncated) and steps < cap:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1

    print("=" * 78)
    print("  INFERENCE RESULT")
    print("=" * 78)
    print(f"  model            : {model_path}")
    print(f"  image            : {image_path}")
    print(f"  start point      : {'random curriculum spawn' if start is None else start}")
    print(f"  deterministic    : {deterministic}")
    print("-" * 78)
    print(f"  coverage         : {info['coverage_pct']:.2f}%")
    print(f"  covered pixels   : {info['covered_pixels']} / {info['total_contour_pixels']}")
    print(f"  total reward     : {total_reward:+.2f}")
    print(f"  steps            : {info['steps']}")
    print(f"  task complete    : {info['task_complete']}")
    print(f"  terminated       : {terminated}")
    print(f"  truncated        : {truncated}")
    print("=" * 78)

    if output_path is None:
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        tag = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(config.RESULTS_DIR, f"inference_{tag}.png")
    env.render_coverage(
        save_path=output_path,
        title=f"Inference -- coverage {info['coverage_pct']:.2f}%",
    )
    print(f"Saved coverage visualization to {output_path}")

    env.close()
    return info, total_reward, steps


def parse_args():
    p = argparse.ArgumentParser(description="Run a trained PPO model on BeadEnv.")
    p.add_argument("--model", type=str, default=config.FINAL_MODEL_PATH,
                    help="Path to a trained model .zip.")
    p.add_argument("--image", type=str, default=config.DEFAULT_IMAGE_PATH,
                    help="Path to the target image.")
    p.add_argument("--start", type=str, default=None,
                    help="Fixed start point as 'x,y' pixel coordinates in the "
                         "800x800 working raster (default: random curriculum spawn).")
    p.add_argument("--stochastic", action="store_true",
                    help="Use stochastic (sampled) actions instead of the "
                         "deterministic policy mean.")
    p.add_argument("--max-steps", type=int, default=None,
                    help="Per-episode step cap (default: the environment's own max_steps).")
    p.add_argument("--output", type=str, default=None,
                    help="Path to save the coverage visualization "
                         "(default: results/inference_<timestamp>.png).")
    return p.parse_args()


def main():
    args = parse_args()
    run_inference(
        model_path=args.model,
        image_path=args.image,
        start=parse_start(args.start),
        deterministic=not args.stochastic,
        max_steps=args.max_steps,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
