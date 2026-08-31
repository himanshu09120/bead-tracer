"""
benchmark/baseline_theile_uav/grid_env.py

Reimplementation of the environment/MDP in:

    Theile, M., Bayerlein, H., Nai, R., Gesbert, D., Caccamo, M. (2020).
    "UAV Coverage Path Planning under Varying Power Constraints using Deep
    Reinforcement Learning." 2020 IEEE/RSJ International Conference on
    Intelligent Robots and Systems (IROS). DOI: 10.1109/IROS45743.2020.9340934.
    (arXiv: 2003.02609)

NO OFFICIAL CODE is available for THIS specific paper. Note on a related
repository: github.com/theilem/uavSim exists and is maintained by the same
lab, but its current code implements LATER, DIFFERENT papers in the same
research lineage ("Learning to Recharge", arXiv:2309.03157, and "Equivariant
Ensembles...", arXiv:2403.12856); its README points to an "icar" branch for
an even different (ICAR conference) paper. None of these correspond to the
exact 2020 IROS paper cited above, so using that repository here would
misattribute results. This is therefore a faithful REIMPLEMENTATION from the
2020 paper's own methodology section, not run from any downloaded code.

Per the paper's Section II/III (exact figures/algorithm quoted in the
report):
  - N x N grid with a 3-channel map: (1) start/landing zones, (2) target
    zones (must be covered), (3) no-fly zones.
  - Coverage grid: boolean, one cell per grid cell, marks whether a target
    cell has been seen by the UAV's fixed 3x3 field of view.
  - State also includes: UAV position (fed as a one-hot 2D map), remaining
    movement budget (scalar), and a safety flag.
  - Action space: {north, east, south, west, land} (Discrete(5)).
  - Movement budget: sampled uniformly per episode from a fixed range
    (paper uses 25-75); EVERY action (accepted or rejected) costs 1 unit.
  - Reward components (paper defines these symbolically as r_cov, r_sc,
    r_mov, r_crash but does NOT give numeric magnitudes anywhere in the
    text -- an ASSUMPTION we must fill in and document, not the paper's own
    published values): r_cov=+1.0 per newly covered target cell,
    r_sc=-0.5 if the safety controller rejects the action (attempted
    no-fly-zone entry), r_mov=-0.05 per action (constant movement cost),
    r_crash=-10.0 if the movement budget reaches 0 without a safe landing.
  - Terminates on landing in a landing zone OR movement budget reaching 0.

ALGORITHM NOTE: the paper trains a Double DQN (DDQN) -- action SELECTION
from the online network, action VALUE from the target network. SB3's
off-the-shelf DQN (used here, since it is the closest available library
implementation and this benchmark otherwise uses SB3 throughout) computes
the standard, non-double DQN target (max over the target network directly).
This reimplementation therefore trains VANILLA DQN, not DDQN -- a real,
documented algorithmic simplification, not the paper's exact method.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# North, East, South, West, Land
_MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1), None]


class TheileUAVCoverageEnv(gym.Env):
    """Grid-based UAV coverage-under-power-constraint environment, per
    Theile, Bayerlein, Nai, Gesbert & Caccamo, IEEE/RSJ IROS 2020."""

    metadata = {"render_modes": []}

    GRID_SIZE = 16
    FOV = 3  # 3x3 field of view, per the paper
    BUDGET_RANGE = (25, 75)  # per the paper's evaluation setup

    R_COV = 1.0
    R_SC = -0.5
    R_MOV = -0.05
    R_CRASH = -10.0

    def __init__(self, layout_seed: int = 0):
        super().__init__()
        self.layout_seed = int(layout_seed)
        g = self.GRID_SIZE

        self.action_space = spaces.Discrete(5)
        # Flattened observation for SB3 MlpPolicy: 3-channel map + coverage
        # grid + one-hot position map (all g*g) + normalized movement budget.
        obs_dim = g * g * 3 + g * g + g * g + 1
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self._build_map()
        self.rng = np.random.default_rng(layout_seed)

    def _build_map(self):
        """Builds a fixed 3-channel map for this layout_seed: a rectangular
        start/landing zone in one corner, a target region in the middle,
        and a no-fly border/obstacle -- a simplified analog of the paper's
        hand-designed Maps A/B/C, since their exact pixel layouts are not
        published, only shown as figures."""
        rng = np.random.default_rng(self.layout_seed)
        g = self.GRID_SIZE
        self.start_land = np.zeros((g, g), dtype=bool)
        self.target = np.zeros((g, g), dtype=bool)
        self.no_fly = np.zeros((g, g), dtype=bool)

        self.start_land[0:3, 0:3] = True
        self.no_fly[0, :] = True
        self.no_fly[-1, :] = True
        self.no_fly[:, 0] = True
        self.no_fly[:, -1] = True
        # A few random rectangular no-fly obstacles inside the interior.
        for _ in range(3):
            r0 = rng.integers(2, g - 4)
            c0 = rng.integers(2, g - 4)
            h, w = rng.integers(1, 3), rng.integers(1, 3)
            self.no_fly[r0:r0 + h, c0:c0 + w] = True
        self.target[3:g - 1, 3:g - 1] = True
        self.target[self.no_fly] = False
        self.target[self.start_land] = False
        self.total_target_cells = max(int(self.target.sum()), 1)

    def _get_obs(self) -> np.ndarray:
        g = self.GRID_SIZE
        pos_map = np.zeros((g, g), dtype=np.float32)
        pos_map[self._pos] = 1.0
        m = np.stack([self.start_land, self.target, self.no_fly], axis=-1).astype(np.float32)
        return np.concatenate([
            m.flatten(),
            self._covered.astype(np.float32).flatten(),
            pos_map.flatten(),
            [self._budget / self.BUDGET_RANGE[1]],
        ]).astype(np.float32)

    def _mark_fov(self):
        r, c = self._pos
        h = self.FOV // 2
        r0, r1 = max(0, r - h), min(self.GRID_SIZE, r + h + 1)
        c0, c1 = max(0, c - h), min(self.GRID_SIZE, c + h + 1)
        new = self.target[r0:r1, c0:c1] & ~self._covered[r0:r1, c0:c1]
        n_new = int(new.sum())
        self._covered[r0:r1, c0:c1] |= self.target[r0:r1, c0:c1]
        return n_new

    def coverage_info(self) -> dict:
        covered = int(self._covered.sum())
        pct = 100.0 * covered / self.total_target_cells
        return {
            "coverage_pct": pct,
            "covered_cells": covered,
            "total_target_cells": self.total_target_cells,
            "task_complete": covered >= self.total_target_cells,
            "steps": self.step_count,
            "budget": self._budget,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        starts = np.argwhere(self.start_land)
        idx = self.rng.integers(0, len(starts))
        self._pos = tuple(starts[idx])
        self._covered = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=bool)
        self._budget = int(self.rng.integers(self.BUDGET_RANGE[0], self.BUDGET_RANGE[1] + 1))
        self.step_count = 0
        self._landed = False
        self._mark_fov()
        return self._get_obs(), self.coverage_info()

    def step(self, action: int):
        move = _MOVES[int(action)]
        reward = self.R_MOV
        self.step_count += 1
        self._budget -= 1

        if move is None:  # land
            if self.start_land[self._pos]:
                self._landed = True
        else:
            dr, dc = move
            r, c = self._pos
            nr, nc = r + dr, c + dc
            out = nr < 0 or nr >= self.GRID_SIZE or nc < 0 or nc >= self.GRID_SIZE
            if out or self.no_fly[nr, nc]:
                reward += self.R_SC  # safety controller rejects the move
            else:
                self._pos = (nr, nc)
                n_new = self._mark_fov()
                reward += self.R_COV * n_new

        info = self.coverage_info()
        info["landed"] = self._landed
        terminated = self._landed
        truncated = False
        if not terminated and self._budget <= 0:
            reward += self.R_CRASH
            truncated = True

        return self._get_obs(), reward, terminated, truncated, info

    def get_row_col(self):
        return self._pos

    def close(self):
        pass
