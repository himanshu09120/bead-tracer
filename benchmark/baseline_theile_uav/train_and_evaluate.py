"""
benchmark/baseline_theile_uav/train_and_evaluate.py

Trains a (vanilla, see grid_env.py docstring) DQN on TheileUAVCoverageEnv
and evaluates it with this benchmark's common metrics.

Usage:
    python benchmark/baseline_theile_uav/train_and_evaluate.py --device cuda
"""

import argparse
import os
import sys
import time

import torch

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "benchmark"))

from stable_baselines3 import DQN  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.utils import set_random_seed  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402
import stable_baselines3  # noqa: E402

from grid_env import TheileUAVCoverageEnv  # noqa: E402
from common.metrics import (  # noqa: E402
    EpisodeResult, aggregate_results, path_length_from_positions,
    redundancy_rate_from_productive_steps, save_results,
)

# Paper's own Table I hyperparameters, used where SB3's DQN exposes an
# equivalent knob; N_max=10,000 episodes translates here to a timestep
# budget (episodes vary in length with the movement-budget sampling, so we
# use a fixed total-timestep budget instead, documented as our own choice).
TIMESTEPS = 500_000
TRAIN_SEED = 42
N_EVAL_LAYOUTS = 20
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def make_env(layout_seed: int):
    def _init():
        return Monitor(TheileUAVCoverageEnv(layout_seed=layout_seed))
    return _init


def train(device: str) -> dict:
    set_random_seed(TRAIN_SEED)
    vec_env = DummyVecEnv([make_env(TRAIN_SEED)])

    model = DQN(
        "MlpPolicy", vec_env, seed=TRAIN_SEED, device=device, verbose=0,
        buffer_size=50_000,          # paper: |D| = 50,000
        gamma=0.95,                   # paper: gamma = 0.95
        tau=0.005,                    # paper: tau = 0.005 (soft target update)
        batch_size=128,               # paper: m = 128
        learning_rate=1e-4,           # not specified by the paper; SB3-typical value used
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
    )

    t0 = time.time()
    model.learn(total_timesteps=TIMESTEPS, progress_bar=True)
    wall = time.time() - t0

    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, "dqn_theile.zip")
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
        env = TheileUAVCoverageEnv(layout_seed=layout_seed)
        obs, info = env.reset(seed=layout_seed)

        positions = [tuple(int(v) for v in env.get_row_col())]
        productive_flags = []
        prev_covered = info["covered_cells"]
        total_reward = 0.0

        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += float(reward)
            positions.append(tuple(int(v) for v in env.get_row_col()))
            productive_flags.append(info["covered_cells"] > prev_covered)
            prev_covered = info["covered_cells"]

        episodes.append(EpisodeResult(
            method="theile_dqn",
            episode_index=i,
            seed=layout_seed,
            start_point=list(positions[0]),
            coverage_pct=info["coverage_pct"],
            success=bool(info["landed"] and info["task_complete"]),
            path_length=path_length_from_positions(positions),
            steps=info["steps"],
            redundancy_rate=redundancy_rate_from_productive_steps(productive_flags),
            reward=total_reward,
            extra={
                "covered_cells": info["covered_cells"],
                "total_target_cells": info["total_target_cells"],
                "landed": bool(info["landed"]),
                "remaining_budget": info["budget"],
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            },
        ))
        env.close()
        print(f"  [theile_dqn] [{i + 1:>2}/{N_EVAL_LAYOUTS}] layout_seed={layout_seed} "
              f"coverage={info['coverage_pct']:6.2f}%  steps={info['steps']:>4}  "
              f"landed={info['landed']}  task_complete={info['task_complete']}")

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

    print(f"\n{'=' * 78}\nTraining DQN on TheileUAVCoverageEnv (device={device})\n{'=' * 78}")
    train_meta = train(device)
    print(f"Trained in {train_meta['training_wall_clock_seconds']:.1f}s ({TIMESTEPS:,} timesteps)")

    print(f"\nEvaluating over {N_EVAL_LAYOUTS} layouts...")
    episodes, summary = evaluate(train_meta["model_path"], device)

    meta = {
        "method": "theile_dqn",
        "paper": "Theile, Bayerlein, Nai, Gesbert & Caccamo (2020), 'UAV "
                 "Coverage Path Planning under Varying Power Constraints "
                 "using Deep Reinforcement Learning', IEEE/RSJ IROS 2020, "
                 "DOI 10.1109/IROS45743.2020.9340934",
        "scope_note": "The paper trains Double DQN (DDQN); SB3's off-the-shelf "
                       "DQN computes the standard (non-double) target, so this "
                       "reproduces vanilla DQN, not DDQN. See grid_env.py docstring.",
        "code_source": "REIMPLEMENTATION by us -- no official code exists for "
                        "this specific 2020 paper. A related repo "
                        "(github.com/theilem/uavSim) implements later, "
                        "different papers by the same lab -- not used here to "
                        "avoid misattribution.",
        "environment": "16x16 grid, 3-channel map (start/land, target, no-fly) "
                        "+ coverage grid + position + movement budget, "
                        "reimplemented per the paper's Section II/Fig. 2",
        "assumptions": [
            "Reward magnitudes (r_cov, r_sc, r_mov, r_crash) are not given "
            "numerically anywhere in the paper (only defined symbolically); "
            "we used r_cov=+1.0, r_sc=-0.5, r_mov=-0.05, r_crash=-10.0, "
            "documented as our own choice, not the paper's published values",
            "Exact map layouts (paper's Maps A/B/C) are only shown as figures, "
            "not published as data; a simplified random rectangular-obstacle "
            "layout generator is used instead, seeded per evaluation episode",
            "Training timestep budget (500,000) is our own choice; the paper "
            "specifies N_max=10,000 EPISODES, not a fixed timestep count",
        ],
        "authors_published_results_for_reference_only": {
            "note": "PAPER'S OWN reported numbers, NOT run by us -- landing "
                    "ratio and coverage-vs-budget curves from their Table II "
                    "and Fig. 4",
            "landing_ratio_map_A": "99.37%",
            "landing_ratio_map_B": "99.78%",
            "landing_ratio_map_C": "98.26%",
        },
        **train_meta,
        "eval_seeds": list(range(1000, 1000 + N_EVAL_LAYOUTS)),
    }
    json_path, csv_path = save_results(args.out_dir, "theile_dqn", episodes, summary, meta)
    print(f"\n[theile_dqn] summary: mean_coverage={summary['mean_coverage_pct']:.2f}% "
          f"success_rate={summary['success_rate']*100:.1f}% "
          f"mean_redundancy={summary['mean_redundancy_rate']:.3f}")
    print(f"[theile_dqn] saved: {json_path}, {csv_path}")


if __name__ == "__main__":
    main()
