"""
simulate.py -- Animated replay of a trained PPO agent tracing BeadEnv's
target contour, for live demonstration purposes (e.g. showing a professor
the agent actually working, not just a static coverage picture).

DOES NOT MODIFY the environment, observation/action space, reward function,
PPO model, or training logic. Uses the exact same evaluation pipeline as
evaluate.py:
    - train.load_target_image()              (same image loader)
    - BeadEnv(target_image=img, curriculum=False, **config.ENV_KWARGS)
                                               (same env construction)
    - env.eval_start_points(n)                (same deterministic starts)
    - PPO.load(model_path, device="auto")     (same model loading)
    - model.predict(obs, deterministic=...)   (same action selection)
    - env.step(action)                        (same physics/reward/coverage)

Every number this script displays (coverage_pct, covered_pixels,
total_contour_pixels, task_complete, steps) comes straight out of
env.coverage_info() -- the same dict evaluate.py's run_episode() returns.
The covered/uncovered contour pixels drawn on screen come straight out of
env._contour_pixels / env._cov_mask -- the exact same arrays
env.render_coverage() itself draws from. No coverage metric is recomputed
or approximated here.

Panda3D note: env.py's own render() method references a `contour.py` /
BeadSimulation module that does not exist in this repository, and panda3d
is not installed -- that code path has never been functional here. This
script instead uses matplotlib's animation support (already a project
dependency), matching render_coverage()'s existing dark/grey/green/orange
visual language, so no new heavyweight engine is introduced for what is a
2-D contour-tracing task.

Usage:
    python simulate.py
    python simulate.py --image target_images/path2.png --point 3
    python simulate.py --model models/bead_ppo_final.zip --speed 2 --fps 60
    python simulate.py --stochastic --max-steps 800
    python simulate.py --save results/simulate_path1.gif

Controls (in the matplotlib window):
    SPACE     pause / resume
    R         restart the same episode from the same start point
    Q / ESC   quit
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from stable_baselines3 import PPO

import config
from env import BeadEnv
from train import load_target_image

# Same BGR colors env.render_coverage() uses, converted to RGB hex so the
# live animation matches the project's existing static renders exactly.
BG_COLOR       = "#141414"   # canvas (20, 20, 20) BGR == RGB (equal channels)
UNCOVERED_COLOR = "#696969"  # (105, 105, 105) BGR == RGB (equal channels)
COVERED_COLOR   = "#3ceb3c"  # BGR (60, 235, 60)  -> RGB (60, 235, 60)
TRAJ_COLOR      = "#ffa500"  # BGR (0, 165, 255)  -> RGB (255, 165, 0) orange
START_COLOR     = "#00e6ff"  # BGR (255, 230, 0)  -> RGB (0, 230, 255) cyan
BEAD_COLOR      = "#ff3232"  # BGR (0, 0, 255)    -> RGB (255, 0, 0)   red
TEXT_COLOR      = "#f0f0f0"


# =============================================================================
# Episode capture -- runs the SAME loop as evaluate.py's run_episode(), only
# additionally recording per-step snapshots for animated playback.
# =============================================================================

def run_episode_capture(env: BeadEnv, model: PPO, start_point, deterministic: bool = True,
                         max_steps: int = None):
    """Runs one full episode from a fixed contour start point -- identical
    control flow to evaluate.run_episode() -- and records a frame per step
    for animation. Nothing about coverage/reward/physics is recomputed: every
    recorded field is read directly off the env's own public state
    (coverage_info(), get_pixel_pos(), distance_map_raw) or the exact arrays
    render_coverage() draws from (_contour_pixels, _cov_mask, _traj)."""
    obs, info = env.reset(seed=None, options={"fixed_start": start_point})
    cap = max_steps if max_steps is not None else env.max_steps

    def snapshot(terminated: bool, truncated: bool) -> dict:
        px, py = env.get_pixel_pos()
        frame = dict(info)
        frame["px"] = px
        frame["py"] = py
        frame["dist_to_contour"] = float(env.distance_map_raw[py, px])
        frame["cov_mask"] = env._cov_mask.copy()
        frame["terminated"] = terminated
        frame["truncated"] = truncated
        return frame

    frames = [snapshot(False, False)]

    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < cap:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        frames.append(snapshot(terminated, truncated))

    # env._traj is append-only and now holds the full episode's trajectory;
    # frame i's trajectory prefix is traj[:i + 1] (reset() appends the spawn
    # point once, step() appends one point per call).
    traj = list(env._traj)
    return frames, traj


# =============================================================================
# Animation
# =============================================================================

def build_animation(env: BeadEnv, frames: list, traj: list, image_path: str,
                     model_path: str, deterministic: bool, fps: int, speed: float):
    S = env.IMG_SIZE
    cx = env._contour_pixels[:, 0]
    cy = env._contour_pixels[:, 1]

    fig, ax = plt.subplots(figsize=(8, 8.6))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, S)
    ax.set_ylim(S, 0)  # image coordinates: y grows downward
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    mode = "deterministic" if deterministic else "stochastic"
    ax.set_title(f"BeadEnv simulation -- {image_path}  ({mode} policy)",
                 color=TEXT_COLOR, fontsize=11, pad=10)

    # Target contour (grey, static) drawn first, covered contour (green)
    # drawn on top -- same layering as render_coverage().
    ax.scatter(cx, cy, s=2, c=UNCOVERED_COLOR, marker=".", linewidths=0)
    covered_scatter = ax.scatter([], [], s=2, c=COVERED_COLOR, marker=".", linewidths=0)

    traj_line, = ax.plot([], [], color=TRAJ_COLOR, lw=1.3, alpha=0.9)
    start_px, start_py = frames[0]["px"], frames[0]["py"]
    ax.scatter([start_px], [start_py], s=70, c=START_COLOR, edgecolors="black",
               linewidths=0.5, zorder=5, label="start")
    bead_dot, = ax.plot([], [], marker="o", markersize=11, color=BEAD_COLOR,
                         markeredgecolor="white", markeredgewidth=1.2, zorder=6)

    legend_handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=UNCOVERED_COLOR,
                    markersize=10, label="target contour"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=COVERED_COLOR,
                    markersize=10, label="covered contour"),
        plt.Line2D([0], [0], color=TRAJ_COLOR, lw=2, label="trajectory"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=START_COLOR,
                    markersize=9, label="start"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=BEAD_COLOR,
                    markersize=9, label="bead (live)"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", facecolor="#1e1e1e",
              edgecolor="#444444", labelcolor=TEXT_COLOR, fontsize=8, framealpha=0.85)

    info_text = ax.text(
        0.015, 0.985, "", transform=ax.transAxes, va="top", ha="left",
        color=TEXT_COLOR, fontsize=9.5, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e1e1e", edgecolor="#444444", alpha=0.85),
    )

    # --- playback control state -------------------------------------------------
    state = {"paused": False, "frame_idx": 0, "done_hold": False}

    def info_lines(f: dict) -> str:
        status = "RUNNING"
        if f["terminated"]:
            status = "COMPLETE (80% threshold reached)" if f["task_complete"] else "TERMINATED"
        elif f["truncated"]:
            status = "STOPPED (stagnation / max_steps -- did NOT reach 80%)"
        return (
            f"step            : {f['steps']}\n"
            f"coverage        : {f['coverage_pct']:6.2f}%\n"
            f"covered pixels  : {f['covered_pixels']}/{f['total_contour_pixels']}\n"
            f"dist to contour : {f['dist_to_contour']:6.2f} px\n"
            f"status          : {status}"
        )

    def draw_frame(i: int):
        f = frames[i]
        cp = env._contour_pixels[f["cov_mask"]]
        if len(cp):
            covered_scatter.set_offsets(cp)
        else:
            covered_scatter.set_offsets(np.empty((0, 2)))

        tp = np.asarray(traj[: f["steps"] + 1], dtype=np.float32)
        traj_line.set_data(tp[:, 0], tp[:, 1])
        bead_dot.set_data([f["px"]], [f["py"]])
        info_text.set_text(info_lines(f))
        return covered_scatter, traj_line, bead_dot, info_text

    # `speed` selects which recorded steps get drawn (>1 skips steps to play
    # faster without changing the underlying physics/policy rollout at all);
    # the final frame is always included so the finished state is shown.
    stride = max(int(round(speed)), 1)
    frame_indices = list(range(0, len(frames), stride))
    if frame_indices[-1] != len(frames) - 1:
        frame_indices.append(len(frames) - 1)

    def update(pos: int):
        idx = frame_indices[pos]
        state["frame_idx"] = pos
        return draw_frame(idx)

    ani = FuncAnimation(
        fig, update, frames=len(frame_indices), interval=max(1000 // fps, 1),
        blit=False, repeat=False,
    )

    def on_key(event):
        if event.key == " ":
            if state["paused"]:
                ani.resume()
            else:
                ani.pause()
            state["paused"] = not state["paused"]
        elif event.key in ("r", "R"):
            state["paused"] = False
            ani.frame_seq = ani.new_frame_seq()
            ani.resume()
        elif event.key in ("q", "Q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.tight_layout()
    return fig, ani


# =============================================================================
# Main
# =============================================================================

def print_final_summary(f: dict, image_path: str, model_path: str, deterministic: bool):
    reached_threshold = bool(f["terminated"] and f["task_complete"])
    print("\n" + "=" * 78)
    print("  SIMULATION SUMMARY")
    print("=" * 78)
    print(f"  model                : {model_path}")
    print(f"  image                : {image_path}")
    print(f"  policy               : {'deterministic' if deterministic else 'stochastic'}")
    print("-" * 78)
    print(f"  final coverage       : {f['coverage_pct']:.2f}%  "
          f"({f['covered_pixels']}/{f['total_contour_pixels']} px)")
    print(f"  total steps          : {f['steps']}")
    print(f"  task complete        : {f['task_complete']}")
    print(f"  terminated           : {f['terminated']}")
    print(f"  truncated            : {f['truncated']}")
    print(f"  ended via 80% cutoff : {reached_threshold}")
    print("=" * 78)


def parse_args():
    p = argparse.ArgumentParser(
        description="Animate a trained PPO agent tracing BeadEnv's target contour.")
    p.add_argument("--model", type=str, default=config.FINAL_MODEL_PATH,
                    help="Path to a trained model .zip.")
    p.add_argument("--image", type=str, default=config.DEFAULT_IMAGE_PATH,
                    help="Path to the target image.")
    p.add_argument("--point", type=int, default=0,
                    help="Index into env.eval_start_points(--n-points) to start from "
                         "(same deterministic points evaluate.py uses).")
    p.add_argument("--n-points", type=int, default=config.EVAL_N_START_POINTS,
                    help="How many deterministic start points to choose --point from.")
    p.add_argument("--stochastic", action="store_true",
                    help="Use stochastic (sampled) actions instead of the "
                         "deterministic policy mean.")
    p.add_argument("--max-steps", type=int, default=config.EVAL_MAX_STEPS,
                    help="Per-episode step cap (same default evaluate.py uses).")
    p.add_argument("--fps", type=int, default=60,
                    help="Playback frame rate (frames per second).")
    p.add_argument("--speed", type=float, default=1.0,
                    help="Playback speed multiplier -- e.g. 2 shows every 2nd "
                         "recorded step (faster demo); does not alter the "
                         "underlying physics or policy rollout.")
    p.add_argument("--save", type=str, default=None,
                    help="Optional path to save the animation as a .gif "
                         "instead of (or in addition to) showing it live.")
    p.add_argument("--no-show", action="store_true",
                    help="Skip the interactive window (useful with --save on "
                         "a headless machine).")
    return p.parse_args()


def main():
    args = parse_args()

    img = load_target_image(args.image)
    env = BeadEnv(target_image=img, curriculum=False, **config.ENV_KWARGS)

    model = PPO.load(args.model, device="auto")

    start_points = env.eval_start_points(args.n_points)
    point_idx = args.point % len(start_points)
    start_point = start_points[point_idx]

    print(f"Simulating {args.model}")
    print(f"  image        : {args.image}")
    print(f"  contour px   : {env.total_contour_pixels}")
    print(f"  start point  : #{point_idx} = {start_point} "
          f"(of {len(start_points)} deterministic points)")
    print(f"  policy       : {'stochastic' if args.stochastic else 'deterministic'}")
    print("Running episode...")

    frames, traj = run_episode_capture(
        env, model, start_point,
        deterministic=not args.stochastic,
        max_steps=args.max_steps,
    )

    print_final_summary(frames[-1], args.image, args.model, not args.stochastic)

    fig, ani = build_animation(
        env, frames, traj, args.image, args.model,
        deterministic=not args.stochastic, fps=args.fps, speed=args.speed,
    )

    if args.save:
        print(f"\nSaving animation to {args.save} ...")
        ani.save(args.save, writer="pillow", fps=args.fps)
        print(f"Saved animation to {args.save}")

    if not args.no_show:
        plt.show()

    env.close()


if __name__ == "__main__":
    main()
