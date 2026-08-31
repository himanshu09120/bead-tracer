"""
benchmark/baseline_sensors_actorcritic/grid_env.py

Reimplementation of the environment described in:

    Garrido-Castaneda, S.I., Vasquez, J.I., Antonio-Cruz, M. (2025).
    "Coverage Path Planning Using Actor-Critic Deep Reinforcement Learning."
    Sensors, 25(5), 1592. DOI: 10.3390/s25051592.

NO OFFICIAL CODE IS AVAILABLE for this paper (its Data Availability
Statement reads "Data are available on request" -- verified by fetching the
publisher page directly; no GitHub link is provided anywhere in the text).
This is therefore a REIMPLEMENTATION from the published methodology
description, not the authors' code, and is labeled as such everywhere it is
used in this benchmark.

Per the paper (as described in the Methods section reachable via the public
PMC mirror):
  - Grid world: 9x9 cells.
  - State: robot (row, col) position, 8 boolean sensor readings for
    adjacent-cell obstacles, and the count of covered cells.
  - Action space: Discrete(4) -- move North / East / South / West.
  - Reward: -0.1 per step; +0.1 per newly covered cell; -10 for hitting an
    obstacle/wall (episode continues, agent stays in place); +0.01 for a
    free (non-covering, non-colliding) move; +10 one-time terminal bonus on
    full coverage.
  - Termination: full coverage of all free cells, OR 500 timesteps elapsed.
  - Original paper trains with Stable-Baselines3 (v1.0.8) A2C and PPO.

ASSUMPTION FLAGGED (not specified in the extractable methodology and
therefore an explicit limitation of this reproduction): the paper's exact
obstacle layout/density generation procedure. This environment generates a
configurable random obstacle layout (default density) with a fixed seed per
environment instance so it is fully reproducible run-to-run; this is
DOCUMENTED as an assumption, not presented as the paper's own obstacle
generation method (which not available to us).

Ground-truth coverage is tracked as a boolean mask over every individual
free (non-obstacle) grid cell -- analogous in spirit to BeadEnv's per-pixel
mask -- never inferred from reward or step count.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SensorsGridCoverageEnv(gym.Env):
    """9x9 discrete grid coverage environment, per Garrido-Castaneda et al. 2025."""

    metadata = {"render_modes": []}

    GRID_SIZE = 9
    N_ACTIONS = 4  # N, E, S, W
    MAX_STEPS = 500

    STEP_PENALTY = -0.1
    NEW_CELL_REWARD = 0.1
    COLLISION_PENALTY = -10.0
    FREE_MOVE_REWARD = 0.01
    COMPLETION_BONUS = 10.0

    # (drow, dcol) for N, E, S, W
    _MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def __init__(self, obstacle_density: float = 0.1, layout_seed: int = 0):
        super().__init__()
        self.obstacle_density = float(obstacle_density)
        self.layout_seed = int(layout_seed)

        self.action_space = spaces.Discrete(self.N_ACTIONS)
        # [row_norm, col_norm, 8 sensor bools, covered_frac] = 11 dims,
        # matching the paper's stated state composition (position, sensor
        # reading, number of covered cells) flattened into one Box for
        # SB3's MlpPolicy.
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(11,), dtype=np.float32
        )

        self._build_layout()
        self._pos = None
        self._covered = None
        self.step_count = 0

    def _build_layout(self):
        rng = np.random.default_rng(self.layout_seed)
        g = self.GRID_SIZE
        self.obstacle_mask = rng.random((g, g)) < self.obstacle_density
        # Guarantee at least one free cell and that free cells form a
        # traversable set is not verified here (not specified by the
        # paper); documented as a known simplification.
        self.obstacle_mask[0, 0] = False  # fixed start cell always free
        self.total_free_cells = int((~self.obstacle_mask).sum())

    def _sensor_readings(self, row: int, col: int) -> np.ndarray:
        """8 boolean obstacle/out-of-bounds readings for the Moore
        neighborhood (N, NE, E, SE, S, SW, W, NW), matching the paper's
        '8 rays detecting adjacent obstacles' description."""
        offsets = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                   (1, 0), (1, -1), (0, -1), (-1, -1)]
        readings = np.zeros(8, dtype=np.float32)
        for i, (dr, dc) in enumerate(offsets):
            r, c = row + dr, col + dc
            if r < 0 or r >= self.GRID_SIZE or c < 0 or c >= self.GRID_SIZE:
                readings[i] = 1.0
            elif self.obstacle_mask[r, c]:
                readings[i] = 1.0
        return readings

    def _get_obs(self) -> np.ndarray:
        row, col = self._pos
        sensors = self._sensor_readings(row, col)
        covered_frac = float(self._covered.sum()) / max(self.total_free_cells, 1)
        return np.concatenate([
            [row / (self.GRID_SIZE - 1), col / (self.GRID_SIZE - 1)],
            sensors,
            [covered_frac],
        ]).astype(np.float32)

    def coverage_info(self) -> dict:
        """Ground-truth coverage, read directly from the boolean cell mask
        -- the single source of truth for this environment, analogous to
        BeadEnv.coverage_info()."""
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

        reward = self.STEP_PENALTY
        collided = False
        new_cell = False

        out_of_bounds = nr < 0 or nr >= self.GRID_SIZE or nc < 0 or nc >= self.GRID_SIZE
        if out_of_bounds or (not out_of_bounds and self.obstacle_mask[nr, nc]):
            collided = True
            reward += self.COLLISION_PENALTY
            # Stay in place on collision.
        else:
            self._pos = (nr, nc)
            if not self._covered[nr, nc]:
                self._covered[nr, nc] = True
                new_cell = True
                reward += self.NEW_CELL_REWARD
            else:
                reward += self.FREE_MOVE_REWARD

        self.step_count += 1
        info = self.coverage_info()
        info["collided"] = collided
        info["new_cell"] = new_cell

        terminated = False
        if info["task_complete"]:
            reward += self.COMPLETION_BONUS
            terminated = True

        truncated = (not terminated) and (self.step_count >= self.MAX_STEPS)

        return self._get_obs(), reward, terminated, truncated, info

    def get_row_col(self):
        return self._pos

    def close(self):
        pass
