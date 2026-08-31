"""
benchmark/baseline_devo_entropy/grid_env.py

Reimplementation of the environment/reward in:

    Devo, A., Mao, J., Costante, G., Loianno, G. (2022). "Autonomous
    Single-Image Drone Exploration With Deep Reinforcement Learning and
    Mixed Reality." IEEE Robotics and Automation Letters, 7(2), 5031-5038.
    DOI: 10.1109/LRA.2022.3154019.

SCOPE, STATED UP FRONT: the paper's actual environment is a photorealistic
Unreal Engine 4 simulation with a real drone + Vicon motion-capture mixed-
reality deployment, using raw 84x84 RGB camera frames as the actor's
observation. NONE of that (UE4, a drone, a motion-capture lab) is
reproducible here. Per this project's own rule ("if exact reproduction is
impossible, document the limitation and implement the closest scientifically
valid version"), this file reproduces exactly the two things the paper
specifies in closed form and that ARE reproducible -- the discrete 11-action
space (Fig. 2) and the entropy-based reward (Eqs. 3-4) -- inside a 2-D grid
proxy environment with a LOCAL PARTIAL occupancy crop standing in for the
RGB frame (preserving the paper's core POMDP property: the agent cannot see
the whole map at once), rather than a photorealistic image.

Action space (Discrete(11), exact labels/order from the paper's Fig. 2):
    0 move_forwards            6 turn_left_move_backward
    1 turn_left                7 move_backward
    2 turn_right                8 do_nothing
    3 turn_right_move_forward   9 move_left  (strafe)
    4 turn_left_move_forward   10 move_right (strafe)
    5 turn_right_move_backward
Heading in {N, E, S, W}; turning changes heading without moving; strafing
moves perpendicular to heading without changing it.

Reward (paper Eqs. 3-4): r = max(Me) / (1 + sum_i ceil(Me_i)), where Me is a
per-cell "map entropy" value. Working through the paper's definition: Me_i =
1/(sum of unexplored Me_j) if cell i is unexplored else 0 -- i.e. a uniform
distribution over the u currently-unexplored cells (each = 1/u). This gives
max(Me) = 1/u and sum(ceil(Me_i)) = u for u > 0 (each unexplored cell's
value ceil's to 1), so r = (1/u) / (1+u): a small positive reward every
step, which GROWS as fewer cells remain unexplored -- exactly the property
the paper's own text describes ("the more cells are explored, the more
complex it is to find unvisited cells, and, hence, a greater reward is
earned"). r = 0 once the whole map is explored (u = 0).

Coverage % (this benchmark's ground-truth metric) = explored_cells / total_free_cells * 100.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# heading 0=N,1=E,2=S,3=W ; (drow, dcol) per heading
_HEADING_DELTA = [(-1, 0), (0, 1), (1, 0), (0, -1)]


class DevoEntropyExplorationEnv(gym.Env):
    """2-D grid proxy for Devo et al. (IEEE RA-L 2022)'s exploration task.
    See module docstring for exactly what is and isn't reproduced."""

    metadata = {"render_modes": []}

    GRID_SIZE = 32
    LOCAL_CROP = 9  # odd size, agent-centered partial observation
    MAX_STEPS = 1800  # paper's own per-episode step budget

    def __init__(self, layout_seed: int = 0):
        super().__init__()
        self.layout_seed = int(layout_seed)
        self.action_space = spaces.Discrete(11)
        c = self.LOCAL_CROP
        # Local occupancy/explored crop (2 channels: wall, explored) + heading one-hot(4)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(c * c * 2 + 4,), dtype=np.float32)

        self._build_map()

    def _build_map(self):
        """A maze-like floor of rooms/corridors -- the closest cheap analog
        of the paper's procedurally generated UE4 floor plans (Fig. 4).
        Random rectangular rooms connected by corridors carved into a
        initially-all-wall grid."""
        rng = np.random.default_rng(self.layout_seed)
        g = self.GRID_SIZE
        self.walls = np.ones((g, g), dtype=bool)
        rooms = []
        for _ in range(6):
            h, w = rng.integers(3, 6), rng.integers(3, 6)
            r0 = rng.integers(1, g - h - 1)
            c0 = rng.integers(1, g - w - 1)
            self.walls[r0:r0 + h, c0:c0 + w] = False
            rooms.append((r0 + h // 2, c0 + w // 2))
        for i in range(1, len(rooms)):
            r0, c0 = rooms[i - 1]
            r1, c1 = rooms[i]
            rr0, rr1 = sorted((r0, r1))
            self.walls[rr0:rr1 + 1, c0] = False
            cc0, cc1 = sorted((c0, c1))
            self.walls[r1, cc0:cc1 + 1] = False
        self._start = rooms[0]
        self.total_free_cells = int((~self.walls).sum())

    def _local_crop(self, channel: np.ndarray) -> np.ndarray:
        g = self.GRID_SIZE
        h = self.LOCAL_CROP // 2
        r, c = self._pos
        padded = np.ones((g + 2 * h, g + 2 * h), dtype=channel.dtype)
        padded[h:h + g, h:h + g] = channel
        return padded[r:r + self.LOCAL_CROP, c:c + self.LOCAL_CROP]

    def _get_obs(self) -> np.ndarray:
        wall_crop = self._local_crop(self.walls.astype(np.float32))
        explored_crop = self._local_crop(self._explored.astype(np.float32))
        heading_onehot = np.zeros(4, dtype=np.float32)
        heading_onehot[self._heading] = 1.0
        return np.concatenate([
            wall_crop.flatten(), explored_crop.flatten(), heading_onehot,
        ]).astype(np.float32)

    def coverage_info(self) -> dict:
        explored = int(self._explored.sum())
        pct = 100.0 * explored / self.total_free_cells
        return {
            "coverage_pct": pct,
            "explored_cells": explored,
            "total_free_cells": self.total_free_cells,
            "task_complete": explored >= self.total_free_cells,
            "steps": self.step_count,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._pos = self._start
        self._heading = 0
        self._explored = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=bool)
        self._explored[self._pos] = True
        self.step_count = 0
        return self._get_obs(), self.coverage_info()

    def _try_move(self, dr: int, dc: int):
        r, c = self._pos
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.GRID_SIZE and 0 <= nc < self.GRID_SIZE and not self.walls[nr, nc]:
            self._pos = (nr, nc)

    def step(self, action: int):
        h = self._heading
        if action == 0:  # move_forwards
            dr, dc = _HEADING_DELTA[h]
            self._try_move(dr, dc)
        elif action == 1:  # turn_left
            self._heading = (h - 1) % 4
        elif action == 2:  # turn_right
            self._heading = (h + 1) % 4
        elif action == 3:  # turn_right_move_forward
            self._heading = (h + 1) % 4
            dr, dc = _HEADING_DELTA[self._heading]
            self._try_move(dr, dc)
        elif action == 4:  # turn_left_move_forward
            self._heading = (h - 1) % 4
            dr, dc = _HEADING_DELTA[self._heading]
            self._try_move(dr, dc)
        elif action == 5:  # turn_right_move_backward
            self._heading = (h + 1) % 4
            dr, dc = _HEADING_DELTA[self._heading]
            self._try_move(-dr, -dc)
        elif action == 6:  # turn_left_move_backward
            self._heading = (h - 1) % 4
            dr, dc = _HEADING_DELTA[self._heading]
            self._try_move(-dr, -dc)
        elif action == 7:  # move_backward
            dr, dc = _HEADING_DELTA[h]
            self._try_move(-dr, -dc)
        elif action == 8:  # do_nothing
            pass
        elif action == 9:  # move_left (strafe)
            dr, dc = _HEADING_DELTA[(h - 1) % 4]
            self._try_move(dr, dc)
        elif action == 10:  # move_right (strafe)
            dr, dc = _HEADING_DELTA[(h + 1) % 4]
            self._try_move(dr, dc)

        self._explored[self._pos] = True
        self.step_count += 1

        u = self.total_free_cells - int(self._explored.sum())
        reward = (1.0 / u) / (1.0 + u) if u > 0 else 0.0

        info = self.coverage_info()
        terminated = info["task_complete"]
        truncated = (not terminated) and (self.step_count >= self.MAX_STEPS)
        return self._get_obs(), reward, terminated, truncated, info

    def get_row_col(self):
        return self._pos

    def close(self):
        pass
