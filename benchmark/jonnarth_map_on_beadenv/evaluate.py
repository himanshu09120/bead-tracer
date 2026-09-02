"""
benchmark/jonnarth_map_on_beadenv/evaluate.py

Answers a specific question: "can our trained PPO model run inside Jonnarth
et al.'s MowerEnv?" -- NO, it cannot (see the conversation this script was
written from for the full explanation): the two environments' observation
spaces (our 64x64 local distance-crop + 11-dim state vs. their four 32x32
multi-scale maps + 24-ray lidar) and action spaces (our 2-D holonomic
thrust vs. their 1-D non-holonomic steering) are structurally incompatible
-- SB3 would raise a shape-mismatch error before a single step ran. Forcing
an adapter to paper over that would invent inputs our model was never
trained to interpret, which is exactly the kind of forced, invalid
comparison this benchmark avoids everywhere else.

What THIS script does instead is import one of Jonnarth et al.'s own
benchmark map IMAGES (maps/eval_mowing_9.png from the official rl-cpp repo
-- a room/corridor floor plan with furniture, structurally similar to the
paper's own Fig. 4/6 training scenes) as a TARGET IMAGE for our own,
completely unmodified BeadEnv -- exactly the same thing evaluate_ours.py
does with target_images/path1.png, just with a different source picture.
This is a legitimate cross-domain generalization test ("how does our
contour-tracing agent handle a shape drawn from a different paper's
benchmark set"), run entirely within our own, compatible environment -- NOT
a claim that our model ran inside MowerEnv, and NOT one of this benchmark's
6 formal paper baselines (it has no coverage-percentage counterpart to
compare against, since Jonnarth's own metric is AREA coverage of this map,
not CONTOUR coverage of its wall/furniture edges).

Usage:
    python benchmark/jonnarth_map_on_beadenv/evaluate.py --device cuda
"""

import argparse
import os
import sys

from stable_baselines3 import PPO

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

MAP_IMAGE = os.path.join(os.path.dirname(__file__), "jonnarth_eval_mowing_9.png")


def run_episode(env: BeadEnv, model: PPO, start_point, episode_index: int,
                 method_name: str, deterministic: bool = True, max_steps: int = None):
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
        method=method_name,
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
    p = argparse.ArgumentParser(
        description="Evaluate our BeadEnv PPO model on a target image drawn "
                    "from Jonnarth et al.'s own benchmark maps.")
    p.add_argument("--model", type=str, default=os.path.join(_PROJECT_ROOT, config.FINAL_MODEL_PATH))
    p.add_argument("--image", type=str, default=MAP_IMAGE)
    p.add_argument("--n-points", type=int, default=config.EVAL_N_START_POINTS)
    p.add_argument("--max-steps", type=int, default=config.EVAL_MAX_STEPS)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--tag", type=str, default=None,
                    help="Distinguishes the saved method name/files when "
                        "--image is overridden, so an ad-hoc run against a "
                        "different image doesn't overwrite the formal "
                        "eval_mowing_9.png result. Default: derived from "
                        "the image filename when --image is non-default.")
    p.add_argument("--out-dir", type=str,
                    default=os.path.join(_PROJECT_ROOT, "benchmark", "results"))
    args = p.parse_args()

    img = load_target_image(args.image)
    env = BeadEnv(target_image=img, curriculum=False, **config.ENV_KWARGS)
    model = PPO.load(args.model, device=args.device)

    is_default_image = os.path.abspath(args.image) == os.path.abspath(MAP_IMAGE)
    tag = args.tag or ("" if is_default_image else
                        "_" + os.path.splitext(os.path.basename(args.image))[0])
    method_name = "ppo_ours_on_jonnarth_map" + tag

    start_points = env.eval_start_points(args.n_points)
    print(f"Evaluating our PPO model on image: {args.image}")
    print(f"  contour pixels : {env.total_contour_pixels}")
    print(f"  device         : {args.device}")
    print(f"  method name    : {method_name}")

    episodes = []
    for i, sp in enumerate(start_points):
        r = run_episode(env, model, sp, i, method_name, max_steps=args.max_steps)
        episodes.append(r)
        print(f"  [{i + 1:>3}/{len(start_points)}] start={tuple(r.start_point)!s:<16} "
              f"coverage={r.coverage_pct:6.2f}%  steps={r.steps:>5}  "
              f"redundancy={r.redundancy_rate:.3f}  complete={r.success}")
    env.close()

    summary = aggregate_results(episodes)
    meta = {
        "method": method_name,
        "purpose": "EXPLORATORY cross-domain generalization test, not one of "
                   "this benchmark's formal paper baselines. Answers "
                   "'how does our contour-tracing agent handle a target "
                   "shape drawn from Jonnarth et al.'s own benchmark set', "
                   "NOT 'our model ran inside MowerEnv' (which is not "
                   "possible -- see this script's module docstring for the "
                   "full incompatibility explanation).",
        "source_image": f"{args.image} -- " + (
            "maps/eval_mowing_9.png from the official rl-cpp repo "
            "(github.com/arvijj/rl-cpp), one of Jonnarth et al.'s own "
            "evaluation floor plans for the lawn-mowing task"
            if is_default_image else
            "a non-default image explicitly passed via --image; see the "
            "path itself for provenance (not auto-described here)"),
        "model_path": args.model,
        "environment": "BeadEnv (this project's own, unmodified environment) "
                       "-- Jonnarth et al.'s map image used only as a "
                       "contour-extraction source, not their environment",
        "n_points": args.n_points,
        "device": args.device,
        "completion_frac": config.ENV_KWARGS["completion_frac"],
    }
    json_path, csv_path = save_results(args.out_dir, method_name, episodes, summary, meta)
    print(f"\nsummary: mean_coverage={summary['mean_coverage_pct']:.2f}% "
          f"success_rate={summary['success_rate']*100:.1f}% "
          f"mean_redundancy={summary['mean_redundancy_rate']:.3f}")
    print(f"saved: {json_path}, {csv_path}")


if __name__ == "__main__":
    main()
