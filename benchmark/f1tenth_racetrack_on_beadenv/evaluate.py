"""
benchmark/f1tenth_racetrack_on_beadenv/evaluate.py

Answers: "can our trained PPO model run inside Elgouhary & El-Wakeel's
F1TENTH Pure-Pursuit-tuning environment?" (arXiv:2602.18386, Feb 2026,
"Learning to Tune Pure Pursuit in Autonomous Racing: Joint Lookahead and
Steering-Gain Control with PPO").

NO, it cannot -- same structural reason as every other paper checked in
this benchmark, just a different flavor of mismatch:
  - Their action is (L_d, g): a lookahead DISTANCE and a steering-gain
    MULTIPLIER that parameterize a classical Pure Pursuit controller --
    not a raw motion command. Our action is a raw 2-D thrust vector
    (Box(-1,1,shape=(2,))). Even though both are 2-D continuous Boxes,
    the two numbers mean entirely different things to entirely different
    downstream controllers; there is no meaningful reinterpretation from
    one to the other.
  - Their observation is a 5-dim vector [v, kappa_0, kappa_1, kappa_2,
    delta_kappa] -- scalar speed and curvature-preview taps along a
    precomputed minimum-curvature raceline. Our observation is
    Dict{local_map: (64,64,1), state: (11,)} -- a local image crop plus a
    hand-crafted state vector. Structurally unrelated.
  - Their environment is F1TENTH Gym (vehicle dynamics + LiDAR + a
    Pure-Pursuit control law); ours is BeadEnv (a bead with 2-D thrust
    physics tracing a contour). Different physics entirely.

What THIS script does instead: this paper's own DATASET is the
`f1tenth_racetracks` collection (github.com/f1tenth/f1tenth_racetracks),
specifically the three tracks it trains/evaluates on -- Hockenheim
(training), Montreal and Yas Marina (zero-shot evaluation). Each track's
map PNG is a single continuous closed racing-line loop -- structurally
much closer to our own training images (target_images/path1/2/3.png,
also single continuous closed curves) than the disconnected floor-plan
maps used in the Jonnarth/Theile-2024 exploratory tests earlier in this
benchmark. This script uses these track images as target images for our
own, completely unmodified BeadEnv, exactly like evaluate_ours.py does
with target_images/path1.png.

This is a legitimate cross-domain generalization test run entirely within
our own, compatible environment -- NOT a claim that our model ran inside
F1TENTH Gym or that it is "tuning Pure Pursuit," and NOT one of this
benchmark's formal paper baselines (no shared metric: their metric is lap
time under a vehicle dynamics model; ours is contour coverage of the
track's boundary line).

Usage:
    python benchmark/f1tenth_racetrack_on_beadenv/evaluate.py --track hockenheim --device cuda
    python benchmark/f1tenth_racetrack_on_beadenv/evaluate.py --track montreal --device cuda
    python benchmark/f1tenth_racetrack_on_beadenv/evaluate.py --track yasmarina --device cuda
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

_HERE = os.path.dirname(os.path.abspath(__file__))
TRACKS = {
    "hockenheim": os.path.join(_HERE, "hockenheim.png"),
    "montreal": os.path.join(_HERE, "montreal.png"),
    "yasmarina": os.path.join(_HERE, "yasmarina.png"),
}


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
        method=f"ppo_ours_on_f1tenth_{env._track_name}",
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
        description="Evaluate our BeadEnv PPO model on F1TENTH racetrack "
                    "images from Elgouhary & El-Wakeel (arXiv:2602.18386).")
    p.add_argument("--model", type=str, default=os.path.join(_PROJECT_ROOT, config.FINAL_MODEL_PATH))
    p.add_argument("--track", type=str, default="hockenheim", choices=list(TRACKS.keys()))
    p.add_argument("--n-points", type=int, default=config.EVAL_N_START_POINTS)
    p.add_argument("--max-steps", type=int, default=config.EVAL_MAX_STEPS)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out-dir", type=str,
                    default=os.path.join(_PROJECT_ROOT, "benchmark", "results"))
    args = p.parse_args()

    image_path = TRACKS[args.track]
    img = load_target_image(image_path)
    env = BeadEnv(target_image=img, curriculum=False, **config.ENV_KWARGS)
    env._track_name = args.track
    model = PPO.load(args.model, device=args.device)

    start_points = env.eval_start_points(args.n_points)
    print(f"Evaluating our PPO model on F1TENTH track: {args.track} ({image_path})")
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
    method_name = f"ppo_ours_on_f1tenth_{args.track}"
    meta = {
        "method": method_name,
        "purpose": "EXPLORATORY cross-domain generalization test, not one "
                   "of this benchmark's formal paper baselines. Answers "
                   "'how does our contour-tracing agent handle a real "
                   "racetrack shape from Elgouhary & El-Wakeel's dataset', "
                   "NOT 'our model ran inside F1TENTH Gym or tunes Pure "
                   "Pursuit' (not possible -- see this script's module "
                   "docstring: action space is controller parameters "
                   "(L_d, g) vs. our raw thrust vector, observation is "
                   "scalar curvature features vs. our image+state).",
        "paper": "Elgouhary & El-Wakeel (2026), 'Learning to Tune Pure "
                 "Pursuit in Autonomous Racing: Joint Lookahead and "
                 "Steering-Gain Control with PPO', arXiv:2602.18386",
        "source_image": f"{args.track}_map.png from the f1tenth_racetracks "
                        "dataset (github.com/f1tenth/f1tenth_racetracks) "
                        "this exact paper trains/evaluates on -- a single "
                        "continuous closed racing-line loop, structurally "
                        "similar to our own target_images/path1-3.png",
        "model_path": args.model,
        "environment": "BeadEnv (this project's own, unmodified environment) "
                       "-- the racetrack image used only as a "
                       "contour-extraction source, not F1TENTH Gym",
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
