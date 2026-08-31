"""
benchmark/baseline_boustrophedon/run_classical_baselines.py

Classical (non-learned) coverage baselines, run INSIDE the project's own,
completely unmodified BeadEnv -- same physics, same action space, same
coverage ground truth (env.coverage_info()) as the trained PPO model. This
is the one baseline in this benchmark that is genuinely apples-to-apples
with our own model, because it uses the identical environment.

WHY NOT LITERAL BOUSTROPHEDON: Boustrophedon Cellular Decomposition (Choset
& Pignon 1997/2000), the standard classical baseline in coverage path
planning literature, targets AREA coverage of a 2-D region via a systematic
back-and-forth sweep. This project's task is CONTOUR coverage -- tracing a
1-D curve embedded in a 2-D plane -- for which a literal region-sweep does
not apply (most of the swept area contains no contour at all). Per the
project's own instructions ("if exact reproduction is impossible, document
the limitation and implement the closest scientifically valid version"),
this script implements the two standard classical analogues used in the
literature when a target is curve-like rather than area-like:

  1. RASTER SWEEP  -- the literal spirit of boustrophedon: a fixed,
     contour-agnostic back-and-forth scan of the whole arena, top to
     bottom. This is exactly the "raster-sweep exploit" env.py's own test
     suite (env._run_tests()) already demonstrates loses badly to honest
     tracing -- reproducing it here as a baseline makes that comparison
     numeric and part of the formal benchmark, not just a unit-test
     assertion.

  2. GREEDY NEAREST-UNCOVERED PURSUIT -- a classical reactive controller:
     every step, thrust directly toward the nearest not-yet-covered contour
     pixel. This is the standard "greedy" baseline for curve/point coverage
     problems. Crucially, it uses ONLY information already present in the
     RL agent's own observation vector (obs["state"][4:6], the unit
     direction to the nearest uncovered pixel that env.py's own
     _get_obs() computes) -- so this baseline and the PPO agent see
     exactly the same information at every step; the only difference is
     that PPO's policy is learned and this one is a fixed formula.

Neither controller uses any BeadEnv private attribute -- both only call the
env's public interface (reset, step, eval_start_points, coverage_info) and
read the public "state" field of its own returned observation, exactly like
evaluate.py and inference.py do.

Usage:
    python benchmark/baseline_boustrophedon/run_classical_baselines.py \
        --image target_images/path1.png --n-points 20
"""

import argparse
import os
import sys
import time

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "benchmark"))

import config  # noqa: E402
from env import BeadEnv  # noqa: E402
from train import load_target_image  # noqa: E402
from common.metrics import (  # noqa: E402
    EpisodeResult, aggregate_results, path_length_from_positions,
    redundancy_rate_from_productive_steps, save_results,
)


def raster_action(step_count: int, boundary: float, period_steps: int = 120):
    """Fixed, contour-agnostic raster-scan action: sweep right for
    `period_steps`, nudge down slightly, sweep left for `period_steps`,
    repeat -- the classical boustrophedon "back-and-forth" motion pattern,
    contour-agnostic by construction."""
    cycle = step_count % (2 * period_steps)
    going_right = cycle < period_steps
    # Small downward drift every full cycle so successive passes don't
    # retrace the same horizontal line forever.
    drift_phase = (step_count // (2 * period_steps)) % 2
    ay = 0.15 if drift_phase == 0 else -0.15
    ax = 1.0 if going_right else -1.0
    return np.array([ax, ay], dtype=np.float32)


def greedy_action(obs: dict):
    """Thrust directly toward the nearest uncovered contour pixel, using
    only obs["state"][4:6] -- the same unit direction vector env.py's own
    _get_obs() already exposes to the PPO agent."""
    dir_x, dir_y = float(obs["state"][4]), float(obs["state"][5])
    if dir_x == 0.0 and dir_y == 0.0:
        return np.array([0.0, 0.0], dtype=np.float32)
    return np.array([dir_x, dir_y], dtype=np.float32)


def run_episode(env: BeadEnv, method: str, start_point, episode_index: int,
                 max_steps: int = None):
    obs, info = env.reset(seed=None, options={"fixed_start": start_point})
    cap = max_steps if max_steps is not None else env.max_steps

    positions = [env.get_pixel_pos_f()]
    productive_flags = []
    prev_covered = info["covered_pixels"]

    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < cap:
        if method == "raster":
            action = raster_action(steps, env.boundary)
        elif method == "greedy":
            action = greedy_action(obs)
        else:
            raise ValueError(f"unknown method: {method}")

        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        positions.append(env.get_pixel_pos_f())
        productive_flags.append(info["covered_pixels"] > prev_covered)
        prev_covered = info["covered_pixels"]

    return EpisodeResult(
        method=f"classical_{method}",
        episode_index=episode_index,
        seed=None,
        start_point=[int(start_point[0]), int(start_point[1])],
        coverage_pct=info["coverage_pct"],
        success=bool(info["task_complete"]),
        path_length=path_length_from_positions(positions),
        steps=info["steps"],
        redundancy_rate=redundancy_rate_from_productive_steps(productive_flags),
        reward=None,
        extra={
            "covered_pixels": info["covered_pixels"],
            "total_contour_pixels": info["total_contour_pixels"],
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        },
    )


def main():
    p = argparse.ArgumentParser(description="Run classical CPP baselines inside BeadEnv.")
    p.add_argument("--image", type=str, default=config.DEFAULT_IMAGE_PATH)
    p.add_argument("--n-points", type=int, default=config.EVAL_N_START_POINTS)
    p.add_argument("--max-steps", type=int, default=config.EVAL_MAX_STEPS)
    p.add_argument("--out-dir", type=str,
                    default=os.path.join(_PROJECT_ROOT, "benchmark", "results"))
    args = p.parse_args()

    img = load_target_image(args.image)

    for method in ("raster", "greedy"):
        env = BeadEnv(target_image=img, curriculum=False, **config.ENV_KWARGS)
        start_points = env.eval_start_points(args.n_points)
        episodes = []
        t0 = time.time()
        for i, sp in enumerate(start_points):
            r = run_episode(env, method, sp, i, max_steps=args.max_steps)
            episodes.append(r)
            print(f"  [{method}] [{i + 1:>3}/{len(start_points)}] "
                  f"start={tuple(r.start_point)!s:<16} coverage={r.coverage_pct:6.2f}%  "
                  f"steps={r.steps:>5}  redundancy={r.redundancy_rate:.3f}  "
                  f"complete={r.success}")
        wall = time.time() - t0
        env.close()

        summary = aggregate_results(episodes)
        meta = {
            "method": f"classical_{method}",
            "paper": "Choset & Pignon (1997/2000), Boustrophedon Cellular Decomposition "
                     "-- adapted here for curve/contour coverage (see module docstring "
                     "for why literal area-boustrophedon does not apply)"
                     if method == "raster" else
                     "Classical greedy nearest-uncovered-target pursuit "
                     "(standard reactive baseline for point/curve coverage tasks)",
            "code_source": "implemented in this repo, no external code",
            "environment": "BeadEnv (this project's own, unmodified environment)",
            "image": args.image,
            "n_points": args.n_points,
            "seeds": "deterministic eval_start_points(), no RNG seed needed (no training, no stochastic policy)",
            "training_required": False,
            "wall_clock_seconds": wall,
        }
        json_path, csv_path = save_results(args.out_dir, f"classical_{method}", episodes, summary, meta)
        print(f"\n[{method}] summary: mean_coverage={summary['mean_coverage_pct']:.2f}% "
              f"success_rate={summary['success_rate']*100:.1f}% "
              f"mean_redundancy={summary['mean_redundancy_rate']:.3f}")
        print(f"[{method}] saved: {json_path}, {csv_path}\n")


if __name__ == "__main__":
    main()
