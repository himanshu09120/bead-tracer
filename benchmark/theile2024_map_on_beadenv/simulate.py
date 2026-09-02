"""
benchmark/theile2024_map_on_beadenv/simulate.py

Animated replay of our trained PPO model tracing the contour of Theile et
al.'s (IEEE/RSJ IROS 2024) tum50.png map asset, inside our own unmodified
BeadEnv. Same purpose/scope caveats as evaluate.py in this folder -- read
its module docstring first.

Usage:
    python benchmark/theile2024_map_on_beadenv/simulate.py --device cuda
    python benchmark/theile2024_map_on_beadenv/simulate.py --save out.gif --no-show
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from stable_baselines3 import PPO

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "benchmark"))

import config  # noqa: E402
from env import BeadEnv  # noqa: E402
from train import load_target_image  # noqa: E402

MAP_IMAGE = os.path.join(os.path.dirname(__file__), "theile2024_tum50.png")

BG_COLOR = "#141414"
UNCOVERED_COLOR = "#696969"
COVERED_COLOR = "#3ceb3c"
TRAJ_COLOR = "#ffa500"
START_COLOR = "#00e6ff"
BEAD_COLOR = "#ff3232"
TEXT_COLOR = "#f0f0f0"


def run_episode_capture(env: BeadEnv, model: PPO, start_point, deterministic: bool = True,
                         max_steps: int = None):
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

    return frames, list(env._traj)


def build_animation(env: BeadEnv, frames: list, traj: list, image_path: str,
                     fps: int, speed: float):
    S = env.IMG_SIZE
    cx = env._contour_pixels[:, 0]
    cy = env._contour_pixels[:, 1]

    fig, ax = plt.subplots(figsize=(8, 8.6))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, S)
    ax.set_ylim(S, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(f"BeadEnv PPO on Theile et al. 2024 IROS map -- {os.path.basename(image_path)}",
                 color=TEXT_COLOR, fontsize=10.5, pad=10)

    ax.scatter(cx, cy, s=2, c=UNCOVERED_COLOR, marker=".", linewidths=0)
    covered_scatter = ax.scatter([], [], s=2, c=COVERED_COLOR, marker=".", linewidths=0)
    traj_line, = ax.plot([], [], color=TRAJ_COLOR, lw=1.3, alpha=0.9)
    start_px, start_py = frames[0]["px"], frames[0]["py"]
    ax.scatter([start_px], [start_py], s=70, c=START_COLOR, edgecolors="black",
               linewidths=0.5, zorder=5, label="start")
    bead_dot, = ax.plot([], [], marker="o", markersize=11, color=BEAD_COLOR,
                         markeredgecolor="white", markeredgewidth=1.2, zorder=6)

    legend_handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=UNCOVERED_COLOR, markersize=10, label="target contour"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=COVERED_COLOR, markersize=10, label="covered contour"),
        plt.Line2D([0], [0], color=TRAJ_COLOR, lw=2, label="trajectory"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=START_COLOR, markersize=9, label="start"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=BEAD_COLOR, markersize=9, label="bead (live)"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", facecolor="#1e1e1e",
              edgecolor="#444444", labelcolor=TEXT_COLOR, fontsize=8, framealpha=0.85)

    info_text = ax.text(
        0.015, 0.985, "", transform=ax.transAxes, va="top", ha="left",
        color=TEXT_COLOR, fontsize=9.5, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e1e1e", edgecolor="#444444", alpha=0.85),
    )

    def info_lines(f: dict) -> str:
        status = "RUNNING"
        if f["terminated"]:
            status = "COMPLETE (threshold reached)" if f["task_complete"] else "TERMINATED"
        elif f["truncated"]:
            status = "STOPPED (stagnation / max_steps)"
        return (
            f"map             : tum50.png (Theile et al. 2024 IROS)\n"
            f"step            : {f['steps']}\n"
            f"coverage        : {f['coverage_pct']:6.2f}%\n"
            f"covered pixels  : {f['covered_pixels']}/{f['total_contour_pixels']}\n"
            f"dist to contour : {f['dist_to_contour']:6.2f} px\n"
            f"status          : {status}"
        )

    def draw_frame(i: int):
        f = frames[i]
        cp = env._contour_pixels[f["cov_mask"]]
        covered_scatter.set_offsets(cp if len(cp) else np.empty((0, 2)))
        tp = np.asarray(traj[: f["steps"] + 1], dtype=np.float32)
        traj_line.set_data(tp[:, 0], tp[:, 1])
        bead_dot.set_data([f["px"]], [f["py"]])
        info_text.set_text(info_lines(f))
        return covered_scatter, traj_line, bead_dot, info_text

    stride = max(int(round(speed)), 1)
    frame_indices = list(range(0, len(frames), stride))
    if frame_indices[-1] != len(frames) - 1:
        frame_indices.append(len(frames) - 1)

    state = {"paused": False}

    def update(pos: int):
        return draw_frame(frame_indices[pos])

    ani = FuncAnimation(fig, update, frames=len(frame_indices),
                         interval=max(1000 // fps, 1), blit=False, repeat=False)

    def on_key(event):
        if event.key == " ":
            (ani.resume() if state["paused"] else ani.pause())
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=os.path.join(_PROJECT_ROOT, config.FINAL_MODEL_PATH))
    p.add_argument("--image", type=str, default=MAP_IMAGE)
    p.add_argument("--point", type=int, default=0)
    p.add_argument("--n-points", type=int, default=config.EVAL_N_START_POINTS)
    p.add_argument("--max-steps", type=int, default=config.EVAL_MAX_STEPS)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--speed", type=float, default=2.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save", type=str, default=None)
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    img = load_target_image(args.image)
    env = BeadEnv(target_image=img, curriculum=False, **config.ENV_KWARGS)
    model = PPO.load(args.model, device=args.device)

    start_points = env.eval_start_points(args.n_points)
    point_idx = args.point % len(start_points)
    start_point = start_points[point_idx]

    print(f"Simulating {args.model} on {args.image} (device={args.device})")
    print(f"  contour px  : {env.total_contour_pixels}")
    print(f"  start point : #{point_idx} = {start_point}")

    frames, traj = run_episode_capture(env, model, start_point, max_steps=args.max_steps)

    f = frames[-1]
    print("\nSIMULATION SUMMARY")
    print(f"  final coverage : {f['coverage_pct']:.2f}% ({f['covered_pixels']}/{f['total_contour_pixels']})")
    print(f"  total steps    : {f['steps']}")
    print(f"  task complete  : {f['task_complete']}")
    print(f"  terminated     : {f['terminated']}")
    print(f"  truncated      : {f['truncated']}")

    fig, ani = build_animation(env, frames, traj, args.image, fps=args.fps, speed=args.speed)

    if args.save:
        print(f"\nSaving animation to {args.save} ...")
        ani.save(args.save, writer="pillow", fps=args.fps)
        print(f"Saved to {args.save}")

    if not args.no_show:
        plt.show()

    env.close()


if __name__ == "__main__":
    main()
