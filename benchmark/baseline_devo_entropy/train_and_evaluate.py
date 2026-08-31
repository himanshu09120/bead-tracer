"""
benchmark/baseline_devo_entropy/train_and_evaluate.py

Trains PPO (SB3) on DevoEntropyExplorationEnv and evaluates with this
benchmark's common metrics.

ALGORITHM NOTE: the paper trains with IMPALA (V-trace, asynchronous
actor-critic). Stable-Baselines3 does not ship an IMPALA implementation, and
this benchmark otherwise uses SB3 throughout for consistency; PPO (SB3) is
used here as the closest available on-policy actor-critic substitute --
documented as a real algorithmic substitution, not the paper's own method.

Usage:
    python benchmark/baseline_devo_entropy/train_and_evaluate.py --device cuda
"""

import argparse
import os
import sys
import time

import torch

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "benchmark"))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.utils import set_random_seed  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402
import stable_baselines3  # noqa: E402

from grid_env import DevoEntropyExplorationEnv  # noqa: E402
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
        return Monitor(DevoEntropyExplorationEnv(layout_seed=layout_seed))
    return _init


def train(device: str) -> dict:
    set_random_seed(TRAIN_SEED)
    vec_env = DummyVecEnv([make_env(TRAIN_SEED)])

    # gamma=0.99 per the paper's own Eq. 1 discount factor.
    model = PPO("MlpPolicy", vec_env, seed=TRAIN_SEED, device=device, verbose=0, gamma=0.99)

    t0 = time.time()
    model.learn(total_timesteps=TIMESTEPS, progress_bar=True)
    wall = time.time() - t0

    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, "ppo_devo.zip")
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
    model = PPO.load(model_path, device=device)

    episodes = []
    for i in range(N_EVAL_LAYOUTS):
        layout_seed = 1000 + i
        env = DevoEntropyExplorationEnv(layout_seed=layout_seed)
        obs, info = env.reset(seed=layout_seed)

        positions = [tuple(int(v) for v in env.get_row_col())]
        productive_flags = []
        prev_explored = info["explored_cells"]
        total_reward = 0.0

        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += float(reward)
            positions.append(tuple(int(v) for v in env.get_row_col()))
            productive_flags.append(info["explored_cells"] > prev_explored)
            prev_explored = info["explored_cells"]

        episodes.append(EpisodeResult(
            method="devo_ppo",
            episode_index=i,
            seed=layout_seed,
            start_point=list(positions[0]),
            coverage_pct=info["coverage_pct"],
            success=bool(info["task_complete"]),
            path_length=path_length_from_positions(positions),
            steps=info["steps"],
            redundancy_rate=redundancy_rate_from_productive_steps(productive_flags),
            reward=total_reward,
            extra={
                "explored_cells": info["explored_cells"],
                "total_free_cells": info["total_free_cells"],
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            },
        ))
        env.close()
        print(f"  [devo_ppo] [{i + 1:>2}/{N_EVAL_LAYOUTS}] layout_seed={layout_seed} "
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

    print(f"\n{'=' * 78}\nTraining PPO on DevoEntropyExplorationEnv (device={device})\n{'=' * 78}")
    train_meta = train(device)
    print(f"Trained in {train_meta['training_wall_clock_seconds']:.1f}s ({TIMESTEPS:,} timesteps)")

    print(f"\nEvaluating over {N_EVAL_LAYOUTS} layouts...")
    episodes, summary = evaluate(train_meta["model_path"], device)

    meta = {
        "method": "devo_ppo",
        "paper": "Devo, Mao, Costante & Loianno (2022), 'Autonomous "
                 "Single-Image Drone Exploration With Deep Reinforcement "
                 "Learning and Mixed Reality', IEEE Robotics and Automation "
                 "Letters 7(2):5031-5038, DOI 10.1109/LRA.2022.3154019",
        "scope_note": "The paper's environment is a photorealistic Unreal "
                       "Engine 4 simulation + real-drone mixed-reality "
                       "deployment with raw 84x84 RGB camera observations, "
                       "trained with IMPALA. None of that (UE4, a drone, "
                       "motion capture, IMPALA) is reproduced. This "
                       "reimplementation keeps the paper's exact 11-action "
                       "space and exact entropy-based reward formula (Eqs. "
                       "3-4) inside a lightweight 2-D grid proxy with a "
                       "LOCAL partial occupancy crop standing in for the RGB "
                       "frame, trained with PPO (closest available SB3 "
                       "actor-critic) instead of IMPALA. See grid_env.py "
                       "docstring for full detail.",
        "code_source": "REIMPLEMENTATION by us -- no official code found for this paper",
        "environment": "32x32 procedurally-generated rooms-and-corridors grid "
                        "(simplified analog of the paper's UE4 floor plans), "
                        "9x9 local occupancy+explored crop observation",
        "authors_published_results_for_reference_only": {
            "note": "PAPER'S OWN reported numbers (Table I), NOT run by us",
            "coverage_pct_standard_env": "58.2%",
            "coverage_pct_large_env": "39.3%",
            "coverage_pct_realistic_env_mean_of_6_floors": "~70% (range 57.6-76.6%)",
        },
        **train_meta,
        "eval_seeds": list(range(1000, 1000 + N_EVAL_LAYOUTS)),
    }
    json_path, csv_path = save_results(args.out_dir, "devo_ppo", episodes, summary, meta)
    print(f"\n[devo_ppo] summary: mean_coverage={summary['mean_coverage_pct']:.2f}% "
          f"success_rate={summary['success_rate']*100:.1f}% "
          f"mean_redundancy={summary['mean_redundancy_rate']:.3f}")
    print(f"[devo_ppo] saved: {json_path}, {csv_path}")


if __name__ == "__main__":
    main()
