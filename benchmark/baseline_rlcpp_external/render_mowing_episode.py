"""
benchmark/baseline_rlcpp_external/render_mowing_episode.py

Same idea as render_exploration_episode.py, but for the "mowing" task's
OFFICIAL checkpoints (mowing_tv1 / mowing_tv2 -- both, per their own
agent_parameters.json, are Soft Actor-Critic, NOT PPO -- there is no PPO
checkpoint in this release despite the paper describing a PPO agent; the
released weights are SAC for every task, mowing included), targeting ONE
SPECIFIC named map file instead of whichever map MowerEnv's own
episode-cycling would pick by default.

MowerEnv.reset() does:
    self.filename = self.eval_maps[(self.current_episode - 1) % len(self.eval_maps)]
    self._load_map(self.filename)
where self.eval_maps is just a plain list built once in __init__ via
glob.glob('maps/eval_mowing*'). To force a specific map deterministically we
overwrite that list, right after construction, with a single-element list
containing only the requested file -- this uses the class's own public
data attribute and reset() logic exactly as designed, it just narrows which
map that logic is allowed to pick.

Must be run in the isolated `rlcpp_baseline` conda environment (see
benchmark/README.md).

Usage (from the official_code directory, in the rlcpp_baseline env):
    python ../render_mowing_episode.py --map maps/eval_mowing_7.png --steps 3000 \
        --out ../../results/ppo_jonnarth_mowing_eval_mowing_7.gif
"""

import argparse
import os
import sys

import imageio

sys.path.insert(0, os.getcwd())  # must be run with cwd = official_code/
from rlm.mower_env import MowerEnv  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="weights/weights/mowing_tv1")
    p.add_argument("--map", type=str, required=True,
                    help="Path (relative to official_code/) to the specific "
                        "eval map to force, e.g. maps/eval_mowing_7.png")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--out", type=str, default="mowing_episode.gif")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--buffer-size", type=int, default=1000,
                    help="Override for SB3's off-policy (SAC) replay buffer "
                        "size at load time -- same reasoning as in "
                        "render_exploration_episode.py.")
    args = p.parse_args()

    if not os.path.isfile(args.map):
        raise SystemExit(f"--map file not found: {args.map} (cwd={os.getcwd()})")

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
    env_args.verbose = False
    env_args.metrics_dir = None

    print(f"exploration flag from checkpoint's own env_parameters.json: {env_args.exploration}")

    env = MowerEnv(**vars(env_args))
    env.eval_maps = [args.map]  # force this exact map on every reset()
    print(f"forced eval_maps -> {env.eval_maps}")

    algo = getattr(importlib.import_module("stable_baselines3"), agent_args.algo)
    load_kwargs = {"buffer_size": args.buffer_size} if agent_args.algo != "PPO" else {}
    model = algo.load(os.path.join(args.checkpoint, "agent"), env=env, **load_kwargs)

    obs = env.reset()
    print(f"loaded map: {env.filename}")
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

    print(f"\nEpisode finished: map={args.map}  steps={steps}  "
          f"coverage={100 * env.coverage_in_percent:.2f}%  "
          f"overlap={100 * env.overlap_in_percent:.2f}%  "
          f"collisions={env.num_collisions}  total_reward={total_reward:.2f}")

    imageio.mimsave(args.out, frames, fps=args.fps)
    print(f"Saved {len(frames)} frames to {args.out}")

    env.close()


if __name__ == "__main__":
    main()
