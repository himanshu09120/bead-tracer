"""
benchmark/common/metrics.py -- Shared, method-agnostic evaluation metrics and
result I/O used identically by every baseline AND by our own trained PPO
model's benchmark run. Having exactly one implementation of each metric is
what makes cross-method numbers comparable at all: no baseline gets its own
bespoke, more-flattering definition of "coverage" or "redundancy".

This module does NOT know about BeadEnv, grid worlds, or any particular
paper's environment. Every function takes plain, already-computed per-step
data (positions and a "did this step cover something new" boolean), so it
works for a continuous physics env, a discrete grid env, or a hand-coded
classical planner alike.

Metrics collected (per user's requirement list):
  - coverage percentage            -> from the CALLER's own ground-truth
                                       coverage source (never approximated
                                       here -- see each baseline's own
                                       coverage bookkeeping).
  - success / completion rate      -> caller-supplied boolean per episode.
  - path length                    -> sum of consecutive position deltas,
                                       in the environment's OWN native units
                                       (pixels for BeadEnv, grid cells for
                                       grid worlds). NOT directly comparable
                                       in absolute terms across different
                                       environments -- documented as such in
                                       the report, not hidden.
  - episode steps                  -> caller-supplied.
  - redundant / repeated movement  -> universal definition used everywhere
                                       in this benchmark: the fraction of
                                       steps that did NOT increase coverage
                                       (redundancy_rate = 1 - productive_steps
                                       / total_steps). Directly comparable
                                       across methods because it is a
                                       dimensionless fraction, unlike raw
                                       path length.
  - mean / std across episodes     -> aggregate_results().
  - seeds / start points used      -> caller-supplied, stored verbatim.
  - computational cost             -> caller-supplied training metadata
                                       (wall-clock seconds, timesteps,
                                       device/GPU name) stored verbatim.
"""

import csv
import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class EpisodeResult:
    """One evaluated episode, in a schema shared by every method in this
    benchmark. `extra` holds anything method-specific (e.g. BeadEnv's
    covered_pixels/total_contour_pixels, or a grid env's cells_covered/
    grid_size) that doesn't fit the common fields but is worth keeping for
    transparency/debugging.
    """
    method: str                 # e.g. "ppo_ours", "boustrophedon", "sensors_a2c"
    episode_index: int
    seed: Optional[int]
    start_point: Optional[list]  # [x, y] or [row, col]; None if not applicable
    coverage_pct: float          # ground-truth coverage, 0-100
    success: bool                # reached the method's own completion criterion
    path_length: float           # sum of step-to-step position deltas, native units
    steps: int
    redundancy_rate: float       # fraction of steps that added zero new coverage
    reward: Optional[float] = None
    extra: dict = field(default_factory=dict)


def path_length_from_positions(positions: list) -> float:
    """Sum of Euclidean distances between consecutive (x, y) positions.
    `positions` must be in the environment's own native coordinate units --
    this function does no unit conversion, so callers must not mix units
    across methods when comparing raw path_length numbers."""
    total = 0.0
    for (x0, y0), (x1, y1) in zip(positions, positions[1:]):
        total += math.hypot(x1 - x0, y1 - y0)
    return total


def redundancy_rate_from_productive_steps(productive_flags: list) -> float:
    """`productive_flags` is one bool per step: True if that step increased
    ground-truth coverage. redundancy_rate = fraction of steps that did NOT.
    Returns 0.0 for a zero-length episode (nothing to be redundant about)."""
    if not productive_flags:
        return 0.0
    productive = sum(1 for p in productive_flags if p)
    return 1.0 - (productive / len(productive_flags))


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def aggregate_results(episodes: list) -> dict:
    """Mean/std/min/max over a list of EpisodeResult (or dicts with the same
    keys). Every method in this benchmark is summarized through this exact
    function -- no method computes its own summary statistics."""
    def get(ep, key):
        return ep[key] if isinstance(ep, dict) else getattr(ep, key)

    coverage = [get(e, "coverage_pct") for e in episodes]
    path_len = [get(e, "path_length") for e in episodes]
    steps = [get(e, "steps") for e in episodes]
    redundancy = [get(e, "redundancy_rate") for e in episodes]
    successes = [bool(get(e, "success")) for e in episodes]
    rewards = [get(e, "reward") for e in episodes if get(e, "reward") is not None]

    summary = {
        "n_episodes": len(episodes),
        "mean_coverage_pct": _mean(coverage),
        "std_coverage_pct": _std(coverage),
        "min_coverage_pct": min(coverage) if coverage else float("nan"),
        "max_coverage_pct": max(coverage) if coverage else float("nan"),
        "success_rate": _mean([1.0 if s else 0.0 for s in successes]),
        "n_success": sum(successes),
        "mean_path_length": _mean(path_len),
        "std_path_length": _std(path_len),
        "mean_steps": _mean(steps),
        "std_steps": _std(steps),
        "mean_redundancy_rate": _mean(redundancy),
        "std_redundancy_rate": _std(redundancy),
    }
    if rewards:
        summary["mean_reward"] = _mean(rewards)
        summary["std_reward"] = _std(rewards)
    return summary


def save_results(out_dir: str, method_name: str, episodes: list, summary: dict,
                  meta: dict):
    """Writes three files under out_dir, all prefixed with method_name:
      <method>.json  -- {meta, summary, episodes} full record
      <method>.csv   -- one row per episode, flat, for spreadsheet/plot use
      (meta and summary are also duplicated inside the JSON's top level for
       convenience; the CSV holds only per-episode rows.)
    `meta` should include at minimum: seeds, and, if training was involved,
    training_timesteps / training_wall_clock_seconds / device.
    """
    os.makedirs(out_dir, exist_ok=True)
    ep_dicts = [asdict(e) if not isinstance(e, dict) else e for e in episodes]

    json_path = os.path.join(out_dir, f"{method_name}.json")
    with open(json_path, "w") as f:
        json.dump({"meta": meta, "summary": summary, "episodes": ep_dicts}, f, indent=2)

    csv_path = os.path.join(out_dir, f"{method_name}.csv")
    if ep_dicts:
        # Flatten `extra` into top-level columns prefixed extra_* for CSV
        # readability; JSON keeps the nested form.
        fieldnames = [k for k in ep_dicts[0].keys() if k != "extra"]
        extra_keys = sorted({k for e in ep_dicts for k in e.get("extra", {})})
        fieldnames += [f"extra_{k}" for k in extra_keys]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for e in ep_dicts:
                row = {k: e[k] for k in e if k != "extra"}
                for k, v in e.get("extra", {}).items():
                    row[f"extra_{k}"] = v
                writer.writerow(row)

    return json_path, csv_path
