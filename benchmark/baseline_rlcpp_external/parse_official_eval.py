"""
benchmark/baseline_rlcpp_external/parse_official_eval.py

Parses the per-episode CSV metrics written by the OFFICIAL rl-cpp eval.py
(Jonnarth, Zhao & Felsberg, ICML 2024 / Jonnarth, Johansson, Zhao & Felsberg,
IEEE Access 2025 -- official code: github.com/arvijj/rl-cpp) into this
benchmark's common EpisodeResult/JSON/CSV schema, so it appears in the same
comparison table as every other method.

Two distinct sources are parsed and saved as two SEPARATE result files,
per the project's requirement to distinguish authors' own numbers from
locally-run numbers:

  1. "rlcpp_mowing_tv1_bundled" -- the reference metrics BUNDLED INSIDE the
     official pretrained-weights download (weights/weights/mowing_tv1/metrics/),
     i.e. numbers the authors themselves computed with their own checkpoint
     and code, shipped as-is. We did not generate these; we only parse them.

  2. "rlcpp_mowing_tv1_local" -- metrics from OUR OWN invocation of the
     official eval.py, using the SAME official checkpoint and code, run on
     this machine (see baseline_rlcpp_external/official_code, isolated
     rlcpp_baseline conda env). This is "we ran the official implementation
     locally", not "the authors' published numbers".

Coverage, path length, and per-step productive/redundant flags are all
computed from the raw per-step CSV columns (coverage fraction, x, y) using
this benchmark's own common/metrics.py functions -- NOT taken from any
pre-aggregated summary -- so the numbers are computed identically to every
other method in this benchmark.

"Success" is defined as final coverage >= 0.99 (eval.py's own default
--goal_coverage max value, i.e. the stricter of its two defaults [0.9, 0.99]).
This is an assumption applied by us during parsing (the actual episode
`done` flag in the CSVs isn't itself recorded); documented here rather than
silently chosen.

Usage:
    python benchmark/baseline_rlcpp_external/parse_official_eval.py
"""

import csv
import glob
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "benchmark"))

from common.metrics import (  # noqa: E402
    EpisodeResult, aggregate_results, path_length_from_positions,
    redundancy_rate_from_productive_steps, save_results,
)

SUCCESS_THRESHOLD = 0.99
_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_PROJECT_ROOT, "benchmark", "results")


def parse_csv(path: str) -> dict:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    positions = [(float(r["x"]), float(r["y"])) for r in rows]
    coverages = [float(r["coverage"]) for r in rows]
    productive_flags = [c1 > c0 for c0, c1 in zip(coverages, coverages[1:])]
    final_coverage = coverages[-1]
    final_length_reported = float(rows[-1]["length"])  # env's own odometry-based length, meters
    final_steps = int(rows[-1]["steps"])
    return {
        "coverage_pct": 100.0 * final_coverage,
        "success": final_coverage >= SUCCESS_THRESHOLD,
        "path_length": path_length_from_positions(positions),  # our own computation, meters
        "path_length_env_reported": final_length_reported,
        "steps": final_steps,
        "redundancy_rate": redundancy_rate_from_productive_steps(productive_flags),
        "collisions": int(float(rows[-1]["collisions"])),
    }


def build_episodes(method_name: str, csv_paths: list) -> list:
    episodes = []
    for i, path in enumerate(sorted(csv_paths)):
        d = parse_csv(path)
        episodes.append(EpisodeResult(
            method=method_name,
            episode_index=i,
            seed=None,
            start_point=None,
            coverage_pct=d["coverage_pct"],
            success=d["success"],
            path_length=d["path_length"],
            steps=d["steps"],
            redundancy_rate=d["redundancy_rate"],
            reward=None,
            extra={
                "path_length_env_reported_m": d["path_length_env_reported"],
                "collisions": d["collisions"],
                "source_csv": os.path.basename(path),
            },
        ))
    return episodes


def save(method_name: str, csv_paths: list, source_label: str):
    if not csv_paths:
        print(f"[{method_name}] no CSV files found, skipping.")
        return
    episodes = build_episodes(method_name, csv_paths)
    summary = aggregate_results(episodes)
    meta = {
        "method": method_name,
        "paper": "Jonnarth, Zhao & Felsberg (2024), 'Learning Coverage Paths in "
                 "Unknown Environments with Deep Reinforcement Learning', ICML "
                 "2024 (PMLR 235:22491-22508); journal extension: Jonnarth, "
                 "Johansson, Zhao & Felsberg (2025), 'Sim-to-Real Transfer of "
                 "Deep Reinforcement Learning Agents for Online Coverage Path "
                 "Planning', IEEE Access 13:106883-106905.",
        "code_source": "OFFICIAL code: github.com/arvijj/rl-cpp (BSD-3-Clause-Clear)",
        "checkpoint_source": "OFFICIAL pretrained weights (mowing_tv1), from the "
                              "Google Drive link in the official README",
        "result_source": source_label,
        "environment": "MowerEnv (official rl-cpp code) -- lawn-mowing task, "
                        "continuous throttle+steering action space, "
                        "multi-scale CNN map observation + lidar",
        "algorithm": "PPO (Stable-Baselines3 1.6.2, as pinned by the official repo)",
        "success_threshold_assumption": f"coverage >= {SUCCESS_THRESHOLD} "
                                         "(eval.py's own default max(--goal_coverage)); "
                                         "applied by us during parsing, not stored in "
                                         "the original CSVs",
        "isolated_environment": "conda env 'rlcpp_baseline', Python 3.9, "
                                 "gym==0.21.0, stable-baselines3==1.6.2, "
                                 "torch==2.1.2+cu121 (see benchmark/README.md)",
    }
    json_path, csv_path = save_results(OUT_DIR, method_name, episodes, summary, meta)
    print(f"[{method_name}] n={summary['n_episodes']} "
          f"mean_coverage={summary['mean_coverage_pct']:.2f}% "
          f"success_rate={summary['success_rate']*100:.1f}% "
          f"mean_redundancy={summary['mean_redundancy_rate']:.3f}")
    print(f"[{method_name}] saved: {json_path}, {csv_path}")


def main():
    code_dir = os.path.join(_HERE, "official_code")

    bundled = glob.glob(os.path.join(
        code_dir, "weights", "weights", "mowing_tv1", "metrics", "eval_metrics_*.csv"))
    save("rlcpp_mowing_tv1_bundled", bundled,
         "Bundled inside the official pretrained-weights download -- computed "
         "by the paper authors with their own checkpoint and code, NOT run by us.")

    local = glob.glob(os.path.join(
        code_dir, "weights", "weights", "mowing_tv1", "metrics_mowing_tv1", "eval_metrics_*.csv"))
    save("rlcpp_mowing_tv1_local", local,
         "Produced by OUR OWN invocation of the official eval.py, using the "
         "same official checkpoint and code, run locally on this machine "
         "(isolated conda env).")


if __name__ == "__main__":
    main()
