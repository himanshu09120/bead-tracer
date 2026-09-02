"""
benchmark/theile2024_map_on_beadenv/evaluate.py

Answers: "can our trained PPO model run inside Theile, Cao, Caccamo &
Sangiovanni-Vincentelli's CPP environment?" -- IEEE/RSJ IROS 2024,
"Equivariant Ensembles and Regularization for Reinforcement Learning in
Map-based Path Planning" (arXiv:2403.12856), official code:
github.com/theilem/uavSim.

NO, it cannot. Their environment (`CPPGym` in src/gym/cpp.py, extending
`GridGym` in src/gym/grid.py) uses:
  - action_space = spaces.Discrete(num_actions)  -- a DISCRETE grid-move
    action space, a different TYPE entirely from our continuous
    Box(-1,1,shape=(2,)) thrust vector (not just a different shape/scale --
    SB3's PPO policy head is a Gaussian over continuous actions and has no
    way to produce or consume a discrete action index).
  - observation_space from a custom `observation_function` -- multi-channel
    map-based Dict tensors (coverage/obstacle/target maps etc., in the same
    architectural family as this lab's earlier Jonnarth et al. work), a
    completely different shape/content from our
    Dict{local_map: (64,64,1), state: (11,)}.
Feeding one into the other raises a shape/type mismatch before a single
step runs. This is the same structural finding as with Jonnarth et al.'s
MowerEnv (see benchmark/jonnarth_map_on_beadenv/): in this research area,
every paper's observation encoding is bespoke to its own network, so true
plug-and-play compatibility between two independently developed papers'
models is essentially never available by construction, not a gap specific
to this one paper.

What THIS script does instead: import one of the official repo's own map
ASSETS (res/tum50.png -- a 50x50 building floor-plan map, one of the real
maps this exact IEEE IROS 2024 paper trains/evaluates on) as a TARGET IMAGE
for our own, completely unmodified BeadEnv -- the same mechanism
evaluate_ours.py uses for target_images/path1.png. The map is pre-upscaled
with NEAREST-neighbor interpolation to 400x400 before being handed to
BeadEnv (see theile2024_tum50.png in this folder) because at its native
50x50 resolution, BeadEnv's own bilinear resize + Gaussian blur (ahead of
Canny edge detection) washes out most of the thin, 1-pixel-wide walls;
nearest-neighbor upscaling preserves their sharp edges so Canny can find
them (this is a preprocessing step on the image only -- BeadEnv's own
pipeline is untouched).

This is a legitimate cross-domain generalization test run entirely within
our own, compatible environment -- NOT a claim that our model ran inside
the official CPPGym, and NOT one of this benchmark's formal paper
baselines (no shared coverage-percentage metric: their coverage is AREA
coverage of free grid cells with a battery/budget constraint; ours is
CONTOUR coverage of the map's wall edges).

Usage:
    python benchmark/theile2024_map_on_beadenv/evaluate.py --device cuda
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

MAP_IMAGE = os.path.join(os.path.dirname(__file__), "theile2024_tum50.png")


def run_episode(env: BeadEnv, model: PPO, start_point, episode_index: int,
                 deterministic: bool = True, max_steps: int = None):
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
        method="ppo_ours_on_theile2024_map",
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
        description="Evaluate our BeadEnv PPO model on a map image from "
                    "Theile et al.'s IEEE IROS 2024 official repo.")
    p.add_argument("--model", type=str, default=os.path.join(_PROJECT_ROOT, config.FINAL_MODEL_PATH))
    p.add_argument("--image", type=str, default=MAP_IMAGE)
    p.add_argument("--n-points", type=int, default=config.EVAL_N_START_POINTS)
    p.add_argument("--max-steps", type=int, default=config.EVAL_MAX_STEPS)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out-dir", type=str,
                    default=os.path.join(_PROJECT_ROOT, "benchmark", "results"))
    args = p.parse_args()

    img = load_target_image(args.image)
    env = BeadEnv(target_image=img, curriculum=False, **config.ENV_KWARGS)
    model = PPO.load(args.model, device=args.device)

    start_points = env.eval_start_points(args.n_points)
    print(f"Evaluating our PPO model on Theile et al. (2024 IROS)'s map image: {args.image}")
    print(f"  contour pixels : {env.total_contour_pixels}")
    print(f"  device         : {args.device}")

    episodes = []
    for i, sp in enumerate(start_points):
        r = run_episode(env, model, sp, i, max_steps=args.max_steps)
        episodes.append(r)
        print(f"  [{i + 1:>3}/{len(start_points)}] start={tuple(r.start_point)!s:<16} "
              f"coverage={r.coverage_pct:6.2f}%  steps={r.steps:>5}  "
              f"redundancy={r.redundancy_rate:.3f}  complete={r.success}")
    env.close()

    summary = aggregate_results(episodes)
    meta = {
        "method": "ppo_ours_on_theile2024_map",
        "purpose": "EXPLORATORY cross-domain generalization test, not one "
                   "of this benchmark's formal paper baselines. Answers "
                   "'how does our contour-tracing agent handle a target "
                   "shape drawn from a real map asset in Theile et al.'s "
                   "IEEE IROS 2024 official repo', NOT 'our model ran "
                   "inside their CPPGym' (not possible -- see this "
                   "script's module docstring for the full incompatibility "
                   "explanation: Discrete vs continuous action space, "
                   "unrelated observation encodings).",
        "paper": "Theile, Cao, Caccamo & Sangiovanni-Vincentelli (2024), "
                 "'Equivariant Ensembles and Regularization for "
                 "Reinforcement Learning in Map-based Path Planning', "
                 "IEEE/RSJ IROS 2024 (arXiv:2403.12856)",
        "source_image": "res/tum50.png from the official uavSim repo "
                        "(github.com/theilem/uavSim), a real building "
                        "floor-plan map this exact paper trains/evaluates "
                        "on, pre-upscaled 50x50 -> 400x400 with "
                        "nearest-neighbor interpolation to preserve sharp "
                        "wall edges through BeadEnv's own Canny pipeline",
        "model_path": args.model,
        "environment": "BeadEnv (this project's own, unmodified environment) "
                       "-- Theile et al.'s map image used only as a "
                       "contour-extraction source, not their environment",
        "n_points": args.n_points,
        "device": args.device,
        "completion_frac": config.ENV_KWARGS["completion_frac"],
    }
    json_path, csv_path = save_results(args.out_dir, "ppo_ours_on_theile2024_map", episodes, summary, meta)
    print(f"\nsummary: mean_coverage={summary['mean_coverage_pct']:.2f}% "
          f"success_rate={summary['success_rate']*100:.1f}% "
          f"mean_redundancy={summary['mean_redundancy_rate']:.3f}")
    print(f"saved: {json_path}, {csv_path}")


if __name__ == "__main__":
    main()
