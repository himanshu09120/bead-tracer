"""
benchmark/baseline_redqn/train_and_evaluate_dqn.py

Trains a vanilla Stable-Baselines3 DQN on the 16x16 grid environment in
grid_env_16.py -- the BASELINE comparator from Chen, Lu, Cui, Luo & Zheng
(2025), NOT their proposed "Re-DQN" enhancement. See grid_env_16.py's module
docstring for the exact scope and why the novel Re-DQN components are not
reproduced.

Hyperparameters use SB3 DQN defaults except where the paper gives an
explicit range (buffer size, gamma, epsilon schedule, learning rate, target
update) -- documented in `meta["assumptions"]` in the saved results, taking
the midpoint/typical value of each stated range since the paper does not
give one fixed value for the baseline DQN specifically.

Usage:
    python benchmark/baseline_redqn/train_and_evaluate_dqn.py
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

from stable_baselines3 import DQN  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.utils import set_random_seed  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402
import stable_baselines3  # noqa: E402

from grid_env_16 import ReDQNBaselineGridEnv  # noqa: E402
from common.metrics import (  # noqa: E402
    EpisodeResult, aggregate_results, path_length_from_positions,
    redundancy_rate_from_productive_steps, save_results,
)

TIMESTEPS = 500_000
TRAIN_SEED = 42
N_EVAL_LAYOUTS = 20
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def make_env(layout_seed: int):
    def _init():
        e = ReDQNBaselineGridEnv(layout_seed=layout_seed)
        return Monitor(e)
    return _init


def train(device: str) -> dict:
    set_random_seed(TRAIN_SEED)
    vec_env = DummyVecEnv([make_env(TRAIN_SEED)])

    model = DQN(
        "MlpPolicy", vec_env, seed=TRAIN_SEED, device=device, verbose=0,
        buffer_size=50_000,           # paper range 5,000-100,000; midpoint-ish
        gamma=0.95,                    # paper range 0.9-0.99
        exploration_initial_eps=1.0,   # paper range start 0.9-1.0
        exploration_final_eps=0.05,    # paper range end 0.01-0.1
        exploration_fraction=0.5,      # spreads the 2,000-10,000-step decay
                                        # range across a meaningful fraction
                                        # of this run's total timesteps
        learning_rate=0.001,           # paper: typically 0.001
        target_update_interval=2000,   # paper range 500-5,000
    )

    t0 = time.time()
    model.learn(total_timesteps=TIMESTEPS, progress_bar=True)
    wall = time.time() - t0

    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, "dqn_grid16.zip")
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


def evaluate(model_path: str, device: str) -> tuple:
    model = DQN.load(model_path, device=device)

    episodes = []
    for i in range(N_EVAL_LAYOUTS):
        layout_seed = 1000 + i
        env = ReDQNBaselineGridEnv(layout_seed=layout_seed)
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
            method="redqn_baseline_dqn",
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
                "end_reason": info.get("end_reason"),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            },
        ))
        env.close()
        print(f"  [dqn] [{i + 1:>2}/{N_EVAL_LAYOUTS}] layout_seed={layout_seed} "
              f"coverage={info['coverage_pct']:6.2f}%  steps={info['steps']:>4}  "
              f"end_reason={info.get('end_reason')}")

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

    print(f"\n{'=' * 78}\nTraining DQN baseline on ReDQNBaselineGridEnv (device={device})\n{'=' * 78}")
    train_meta = train(device)
    print(f"Trained DQN in {train_meta['training_wall_clock_seconds']:.1f}s "
          f"({TIMESTEPS:,} timesteps)")

    print(f"\nEvaluating DQN over {N_EVAL_LAYOUTS} layouts...")
    episodes, summary = evaluate(train_meta["model_path"], device)

    meta = {
        "method": "redqn_baseline_dqn",
        "paper": "Chen, Lu, Cui, Luo & Zheng (2025), 'A Complete Coverage Path "
                 "Planning Algorithm for Lawn Mowing Robots Based on Deep "
                 "Reinforcement Learning', Sensors 25(2):416, DOI 10.3390/s25020416",
        "scope_note": "Reproduces the paper's DISCRETE GRID BASELINE (vanilla DQN) "
                       "only -- NOT the paper's proposed Re-DQN enhancements "
                       "(noisy-linear exploration, dynamic incentive layer, "
                       "curiosity/novelty intrinsic reward, dynamic obstacle-count "
                       "input padding). See grid_env_16.py docstring for why.",
        "code_source": "REIMPLEMENTATION by us -- no official code released "
                        "('The study did not report any data.')",
        "environment": "16x16 discrete grid, reimplemented per paper's stated "
                        "grid size / action space / baseline reward terms",
        "assumptions": [
            "Reward magnitudes use the midpoint of each range the paper reports "
            "for its own hyperparameter search (paper does not give one fixed "
            "baseline-DQN value): P_move=0.05, R_discover=1.0 (magnitude not "
            "given for baseline; our choice), P_obstacle=0.5, R_cc=10.0",
            "Terrain penalty term fixed at zero -- no terrain/height data exists "
            "anywhere in this benchmark",
            "Obstacle density/layout generation procedure not specified by the "
            "paper; a fixed-density random layout per seed is used",
            "DQN hyperparameters (buffer size, gamma, epsilon schedule, target "
            "update interval) set to the midpoint of the paper's stated ranges",
        ],
        "authors_published_baseline_dqn_results_for_reference_only": {
            "note": "These are the PAPER'S OWN reported numbers for their DQN "
                    "baseline, NOT results from running any code locally -- "
                    "quoted here only so the comparison table can show them "
                    "alongside our own measured numbers, clearly labeled.",
            "avg_steps": "~120",
            "avg_tiles_visited_per_episode": "~87",
            "avg_reward": "~65",
        },
        **train_meta,
        "eval_seeds": list(range(1000, 1000 + N_EVAL_LAYOUTS)),
        "eval_start_point": [0, 0],
    }
    json_path, csv_path = save_results(args.out_dir, "redqn_baseline_dqn", episodes, summary, meta)
    print(f"\n[dqn] summary: mean_coverage={summary['mean_coverage_pct']:.2f}% "
          f"success_rate={summary['success_rate']*100:.1f}% "
          f"mean_redundancy={summary['mean_redundancy_rate']:.3f}")
    print(f"[dqn] saved: {json_path}, {csv_path}")


if __name__ == "__main__":
    main()
