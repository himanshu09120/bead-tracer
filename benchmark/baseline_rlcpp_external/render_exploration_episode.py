"""
benchmark/baseline_rlcpp_external/render_exploration_episode.py

Runs ONE episode of Jonnarth et al.'s OFFICIAL "exploration" pretrained
agent (SAC, per this checkpoint's own agent_parameters.json -- NOT PPO;
the mowing-task checkpoints use PPO, exploration uses Soft Actor-Critic)
inside their OWN, unmodified MowerEnv (exploration mode), and renders
it to a GIF using the environment's own built-in `render(mode='rgb_array')`
-- the same renderer that produced the README's exploration_path.png figure
(the composite green=explored/gray=unexplored/white=obstacle/yellow=path/
blue=lidar-ray image).

This is DIFFERENT from every other script in this benchmark: everywhere
else, WE run OUR model, either in our own BeadEnv or (for cross-domain
tests) on an image borrowed from another paper. HERE, we run THEIR model
in THEIR OWN environment, using THEIR OWN official checkpoint
(weights/weights/exploration) -- exactly what their own eval.py/show_path.py
would do, just without iterating over all 23 official eval maps (which
would take far too long for a demo) or requiring a live display.

Must be run in the isolated `rlcpp_baseline` conda environment (see
benchmark/README.md) -- this repo's own gym==0.21.0 / stable-baselines3==1.6.2
stack, incompatible with this project's own Gymnasium/SB3 versions.

Usage (from the official_code directory, in the rlcpp_baseline env):
    python ../render_exploration_episode.py --steps 3000 --out ../../results/exploration_episode.gif
"""

import argparse
import os
import sys

import imageio
import numpy as np

sys.path.insert(0, os.getcwd())  # must be run with cwd = official_code/
from rlm.mower_env import MowerEnv  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="weights/weights/exploration")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--out", type=str, default="exploration_episode.gif")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--buffer-size", type=int, default=1000,
                    help="Override for SB3's off-policy (SAC) replay buffer "
                        "size at load time -- the checkpoint's own saved "
                        "500,000 is wasted memory for pure inference; this "
                        "is exactly what the official eval.py's own "
                        "--buffer_size flag (default 1000) does too.")
    args = p.parse_args()

    import json
    import importlib
    import argparse as ap

    with open(os.path.join(args.checkpoint, "agent_parameters.json")) as f:
        agent_args = ap.Namespace(**json.load(f))
    print(f"algo from checkpoint's own agent_parameters.json: {agent_args.algo}")
    with open(os.path.join(args.checkpoint, "env_parameters.json")) as f:
        env_args = ap.Namespace(**json.load(f))
    env_args.max_episode_steps = args.steps
    env_args.eval = True
    env_args.goal_coverage = 0.99
    env_args.verbose = False
    env_args.metrics_dir = None

    print(f"exploration flag from checkpoint's own env_parameters.json: {env_args.exploration}")

    env = MowerEnv(**vars(env_args))
    algo = getattr(importlib.import_module("stable_baselines3"), agent_args.algo)
    load_kwargs = {"buffer_size": args.buffer_size} if agent_args.algo != "PPO" else {}
    model = algo.load(os.path.join(args.checkpoint, "agent"), env=env, **load_kwargs)

    obs = env.reset()
    frames = [env.render(mode="rgb_array")]
    total_reward = 0.0
    steps = 0
    done = False
    while not done and steps < args.steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        frames.append(env.render(mode="rgb_array"))

    print(f"\nEpisode finished: steps={steps}  "
          f"coverage={100 * env.coverage_in_percent:.2f}%  "
          f"overlap={100 * env.overlap_in_percent:.2f}%  "
          f"collisions={env.num_collisions}  total_reward={total_reward:.2f}")

    imageio.mimsave(args.out, frames, fps=args.fps)
    print(f"Saved {len(frames)} frames to {args.out}")

    env.close()


if __name__ == "__main__":
    main()
