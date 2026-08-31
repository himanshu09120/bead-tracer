"""
benchmark/baseline_redqn/grid_env_16.py

Reimplementation of the BASELINE environment/reward described in:

    Chen, Y., Lu, Z.-M., Cui, J.-L., Luo, H., Zheng, Y.-M. (2025).
    "A Complete Coverage Path Planning Algorithm for Lawn Mowing Robots
    Based on Deep Reinforcement Learning." Sensors, 25(2), 416.
    DOI: 10.3390/s25020416.

SCOPE, STATED UP FRONT: this reproduces the paper's DISCRETE GRID
ENVIRONMENT and its BASELINE reward structure -- the components the paper
itself compares its proposed "Re-DQN" enhancements against (vanilla DQN,
their own reported numbers: ~120 steps, ~87 tiles/episode, ~65 reward). It
does NOT reproduce Re-DQN's novel algorithmic contributions (noisy-linear
exploration layer, a "dynamic incentive" layer, a curiosity/state-novelty
intrinsic reward term, or dynamic-size input padding for a variable
obstacle count). Those are bespoke architectural contributions of the paper,
not available as off-the-shelf library components, and a rough
reimplementation from a methods-section summary risks producing something
inaccurate that would be wrongly attributed to the paper's authors. Per this
project's own rule ("if exact reproduction is impossible, document the
limitation and implement the closest scientifically valid version"), the
closest valid version here is the paper's own DQN baseline comparator,
trained with an off-the-shelf, unmodified SB3 DQN -- clearly NOT "Re-DQN".

Per the paper's Methods section (as extracted from the public PMC mirror):
  - Grid: 16x16 (paper's tested configuration).
  - Action space: Discrete(4) -- up/down/left/right (paper additionally
    specifies a fixed move duration L, which collapses to one grid step per
    action in a discrete-time simulation and is treated as such here).
  - Reward terms reproduced: -Pmove per step, +Rdiscover per newly covered
    tile, +Rcc one-time completion bonus, -Pobstacle on collision.
  - Reward term NOT reproduced: terrain penalty (-Pterrain*max(0,Tdiff)) --
    this project's benchmark has no terrain/height data for any environment,
    so this term is fixed at zero and documented as not applicable, not
    silently dropped.
  - Termination (per the paper): full coverage = success; collision with an
    obstacle = episode ends (failure); reaching the map boundary = episode
    ends (failure). Reproduced exactly as described.

ASSUMPTIONS DOCUMENTED (paper values not specified for the baseline
comparator; the paper gives ranges intended for its own Re-DQN
hyperparameter search, not fixed baseline values):
  - Reward constants use the midpoint of each range the paper reports:
    Pmove=0.05, Rdiscover=1.0 (paper doesn't give a discovery reward
    magnitude directly for the DQN baseline; 1.0 chosen as a standard unit
    reward, documented as our choice), Pobstacle=0.5, Rcc=10.0.
  - Obstacle density/layout generation procedure is not specified; a
    configurable random layout is used, seeded for reproducibility.

Ground-truth coverage is tracked as a boolean mask over free (non-obstacle)
cells, identical in spirit to grid_env.py's SensorsGridCoverageEnv and
BeadEnv's own per-pixel mask.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ReDQNBaselineGridEnv(gym.Env):
    """16x16 discrete grid coverage environment -- baseline DQN comparator
    from Chen, Lu, Cui, Luo & Zheng (2025). See module docstring for exact
    scope and assumptions."""

    metadata = {"render_modes": []}

    GRID_SIZE = 16
    N_ACTIONS = 4
    MAX_STEPS = 500

    P_MOVE = 0.05          # movement penalty per step (paper range 0.01-0.1, midpoint used)
    R_DISCOVER = 1.0        # reward per newly covered tile (magnitude not given for DQN baseline; our choice, documented)
    P_OBSTACLE = 0.5        # collision penalty (paper range 0.1-1.0, midpoint used)
    R_CC = 10.0             # completion bonus

    _MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left

    def __init__(self, obstacle_density: float = 0.15, layout_seed: int = 0):
        super().__init__()
        self.obstacle_density = float(obstacle_density)
        self.layout_seed = int(layout_seed)

        self.action_space = spaces.Discrete(self.N_ACTIONS)
        # [row_norm, col_norm, covered_frac, steps_norm] -- a compact state;
        # the paper's full state additionally includes target/obstacle
        # sub-vectors for its dynamic-input mechanism, which is part of the
        # Re-DQN contribution not reproduced here (see module docstring).
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

        self._build_layout()
        self._pos = None
        self._covered = None
        self.step_count = 0

    def _build_layout(self):
        rng = np.random.default_rng(self.layout_seed)
        g = self.GRID_SIZE
        self.obstacle_mask = rng.random((g, g)) < self.obstacle_density
        self.obstacle_mask[0, 0] = False
        self.total_free_cells = int((~self.obstacle_mask).sum())

    def _get_obs(self) -> np.ndarray:
        row, col = self._pos
        covered_frac = float(self._covered.sum()) / max(self.total_free_cells, 1)
        return np.array([
            row / (self.GRID_SIZE - 1),
            col / (self.GRID_SIZE - 1),
            covered_frac,
            self.step_count / self.MAX_STEPS,
        ], dtype=np.float32)

    def coverage_info(self) -> dict:
        covered = int(self._covered.sum())
        pct = 100.0 * covered / max(self.total_free_cells, 1)
        return {
            "coverage": covered / max(self.total_free_cells, 1),
            "coverage_pct": pct,
            "covered_cells": covered,
            "total_free_cells": self.total_free_cells,
            "task_complete": covered >= self.total_free_cells,
            "steps": self.step_count,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._pos = (0, 0)
        self._covered = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=bool)
        self._covered[0, 0] = True
        self.step_count = 0
        return self._get_obs(), self.coverage_info()

    def step(self, action: int):
        dr, dc = self._MOVES[int(action)]
        row, col = self._pos
        nr, nc = row + dr, col + dc

        reward = -self.P_MOVE
        self.step_count += 1

        out_of_bounds = nr < 0 or nr >= self.GRID_SIZE or nc < 0 or nc >= self.GRID_SIZE
        if out_of_bounds:
            # Paper: reaching the map boundary ends the episode (failure).
            info = self.coverage_info()
            info["end_reason"] = "boundary"
            return self._get_obs(), reward, True, False, info

        if self.obstacle_mask[nr, nc]:
            # Paper: collision with an obstacle ends the episode (failure).
            reward -= self.P_OBSTACLE
            info = self.coverage_info()
            info["end_reason"] = "collision"
            return self._get_obs(), reward, True, False, info

        self._pos = (nr, nc)
        if not self._covered[nr, nc]:
            self._covered[nr, nc] = True
            reward += self.R_DISCOVER

        info = self.coverage_info()
        terminated = False
        if info["task_complete"]:
            reward += self.R_CC
            terminated = True
            info["end_reason"] = "success"
        else:
            info["end_reason"] = None

        truncated = (not terminated) and (self.step_count >= self.MAX_STEPS)
        return self._get_obs(), reward, terminated, truncated, info

    def get_row_col(self):
        return self._pos

    def close(self):
        pass
