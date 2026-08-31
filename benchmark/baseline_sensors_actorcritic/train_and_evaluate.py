"""
benchmark/baseline_sensors_actorcritic/train_and_evaluate.py

Trains A2C and PPO (both via Stable-Baselines3, matching the original
paper's own tooling -- SB3 v1.0.8 in the paper vs. whatever SB3 version this
project's venv has installed, noted in the results metadata) on the grid
environment in grid_env.py, then evaluates both with this benchmark's common
metrics module across multiple independently-seeded grid layouts (playing
the same role evaluate.py's 20 deterministic start points play for BeadEnv
-- here the "varying condition" is the obstacle layout, since start position
is fixed at (0,0) as in the source paper).

ASSUMPTIONS DOCUMENTED (paper details not extractable from the public
methodology summary available to us):
  - Exact training timestep budget: this script uses TIMESTEPS (below),
    chosen to be enough for a 9x9 grid's tiny state/action space to
    converge, not copied from the paper (which was not specified in what
    we could extract).
  - Obstacle density: see grid_env.py's own docstring.
  - Network architecture: SB3's default MlpPolicy (paper did not specify a
    custom architecture in the extractable summary).

GPU: forced device="cuda" per project instruction. Note (not fabricated --
an honest observation, not a paper claim): a 9x9 grid with an 11-dim flat
state and a tiny default MlpPolicy is so small that GPU offers no realistic
speed advantage over CPU for this particular baseline; it is used anyway
for consistency with the rest of this benchmark's training runs.

Usage:
    python benchmark/baseline_sensors_actorcritic/train_and_evaluate.py
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "benchmark"))

from stable_baselines3 import A2C, PPO  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.utils import set_random_seed  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402
import stable_baselines3  # noqa: E402

from grid_env import SensorsGridCoverageEnv  # noqa: E402
from common.metrics import (  # noqa: E402
    EpisodeResult, aggregate_results, path_length_from_positions,
    redundancy_rate_from_productive_steps, save_results,
)

TIMESTEPS = 300_000
TRAIN_SEED = 42
N_EVAL_LAYOUTS = 20
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def make_env(layout_seed: int):
    def _init():
        e = SensorsGridCoverageEnv(layout_seed=layout_seed)
        e = Monitor(e)
        return e
    return _init


def train(algo_name: str, device: str) -> dict:
    set_random_seed(TRAIN_SEED)
    algo_cls = {"a2c": A2C, "ppo": PPO}[algo_name]
    vec_env = DummyVecEnv([make_env(TRAIN_SEED)])

    model = algo_cls("MlpPolicy", vec_env, seed=TRAIN_SEED, device=device, verbose=0)

    t0 = time.time()
    model.learn(total_timesteps=TIMESTEPS, progress_bar=True)
    wall = time.time() - t0

    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, f"{algo_name}_grid.zip")
    model.save(save_path)
    vec_env.close()

    return {
        "model_path": save_path,
        "training_wall_clock_seconds": wall,
        "training_timesteps": TIMESTEPS,
        "device": device,
        "sb3_version": stable_baselines3.__version__,
        "seed": TRAIN_SEED,
    }


def evaluate(algo_name: str, model_path: str, device: str) -> tuple:
    algo_cls = {"a2c": A2C, "ppo": PPO}[algo_name]
    model = algo_cls.load(model_path, device=device)

    episodes = []
    for i in range(N_EVAL_LAYOUTS):
        layout_seed = 1000 + i  # disjoint from training seed
        env = SensorsGridCoverageEnv(layout_seed=layout_seed)
        obs, info = env.reset(seed=layout_seed)

        row, col = env.get_row_col()
        positions = [(row, col)]
        productive_flags = []
        prev_covered = info["covered_cells"]
        total_reward = 0.0

        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += float(reward)
            row, col = env.get_row_col()
            positions.append((row, col))
            productive_flags.append(info["covered_cells"] > prev_covered)
            prev_covered = info["covered_cells"]

        episodes.append(EpisodeResult(
            method=f"sensors_{algo_name}",
            episode_index=i,
            seed=layout_seed,
            start_point=[0, 0],
            coverage_pct=info["coverage_pct"],
            success=bool(info["task_complete"]),
            path_length=path_length_from_positions(positions),
            steps=info["steps"],
            redundancy_rate=redundancy_rate_from_productive_steps(productive_flags),
            reward=total_reward,
            extra={
                "covered_cells": info["covered_cells"],
                "total_free_cells": info["total_free_cells"],
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            },
        ))
        env.close()
        print(f"  [{algo_name}] [{i + 1:>2}/{N_EVAL_LAYOUTS}] layout_seed={layout_seed} "
              f"coverage={info['coverage_pct']:6.2f}%  steps={info['steps']:>4}  "
              f"complete={info['task_complete']}")

    return episodes, aggregate_results(episodes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out-dir", type=str,
                    default=os.path.join(_PROJECT_ROOT, "benchmark", "results"))
    args = p.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: --device cuda requested but CUDA is not available; falling back to cpu.")
        device = "cpu"

    for algo_name in ("a2c", "ppo"):
        print(f"\n{'=' * 78}\nTraining {algo_name.upper()} on SensorsGridCoverageEnv (device={device})\n{'=' * 78}")
        train_meta = train(algo_name, device)
        print(f"Trained {algo_name} in {train_meta['training_wall_clock_seconds']:.1f}s "
              f"({TIMESTEPS:,} timesteps)")

        print(f"\nEvaluating {algo_name.upper()} over {N_EVAL_LAYOUTS} layouts...")
        episodes, summary = evaluate(algo_name, train_meta["model_path"], device)

        meta = {
            "method": f"sensors_{algo_name}",
            "paper": "Garrido-Castaneda, Vasquez & Antonio-Cruz (2025), "
                     "'Coverage Path Planning Using Actor-Critic Deep Reinforcement "
                     "Learning', Sensors 25(5):1592, DOI 10.3390/s25051592",
            "code_source": "REIMPLEMENTATION by us -- no official code released "
                            "('Data are available on request', no GitHub link found)",
            "environment": "9x9 discrete grid, reimplemented per paper's methodology "
                            "description (see grid_env.py docstring for exact assumptions)",
            "assumptions": [
                "obstacle density/layout generation not specified by the paper; "
                "we use a fixed-density random layout per evaluation seed",
                "training timestep budget not specified by the paper; we used "
                f"{TIMESTEPS:,} timesteps",
                "network architecture not specified beyond SB3 usage; we used "
                "SB3's default MlpPolicy",
            ],
            **train_meta,
            "eval_seeds": list(range(1000, 1000 + N_EVAL_LAYOUTS)),
            "eval_start_point": [0, 0],
        }
        json_path, csv_path = save_results(args.out_dir, f"sensors_{algo_name}", episodes, summary, meta)
        print(f"\n[{algo_name}] summary: mean_coverage={summary['mean_coverage_pct']:.2f}% "
              f"success_rate={summary['success_rate']*100:.1f}% "
              f"mean_redundancy={summary['mean_redundancy_rate']:.3f}")
        print(f"[{algo_name}] saved: {json_path}, {csv_path}")


if __name__ == "__main__":
    main()
