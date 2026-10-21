"""
Non-learned controllers.  Interface = evaluation.PPOPolicy: reset(env, i), act(obs, env) -> action (2,).
`speed` is a desired cruise speed in world units/s (env.MAX_SPEED = 15).

All velocity controllers use the same law as env._drive_along: thrust = (v_des - v) / (ACCEL*DT),
clipped to the unit disk.  Pixel and world axes are aligned, so a pixel-space direction is a world direction.

Information each controller uses
  random, zero, raster : none (raster reads only the bead's x position)
  greedy               : privileged -- exact nearest uncovered contour pixel + own velocity
  pure_pursuit         : privileged -- ordered contour polylines + which pixels are covered + own position
  crop_greedy          : ONLY obs["local_map"] (64x64 crop) and obs["state"][2:4] (own velocity)
"""
import math

import cv2
import numpy as np


def _thrust(env, dx, dy, speed, vx=None, vy=None):
    vx = env.vel_x if vx is None else vx
    vy = env.vel_y if vy is None else vy
    d = math.hypot(dx, dy)
    wx, wy = (dx / d * speed, dy / d * speed) if d > 1e-9 else (0.0, 0.0)
    ax = (wx - vx) / (env.ACCEL * env.DT)
    ay = (wy - vy) / (env.ACCEL * env.DT)
    m = math.hypot(ax, ay)
    if m > 1.0:
        ax, ay = ax / m, ay / m
    return np.array([ax, ay], dtype=np.float32)


class _Base:
    name = "base"

    def reset(self, env, i):
        pass


class RandomPolicy(_Base):
    name = "random"

    def reset(self, env, i):
        self.rng = np.random.default_rng(i)

    def act(self, obs, env):
        return self.rng.uniform(-1, 1, 2).astype(np.float32)


class ZeroPolicy(_Base):
    name = "zero"

    def act(self, obs, env):
        return np.zeros(2, dtype=np.float32)


class RasterPolicy(_Base):
    """Blind boustrophedon sweep (the reward-hack probe from env.py's self-test): full thrust along x, reverse
    at |x| > 9.3, nudge down for 8 steps."""
    name = "raster"

    def reset(self, env, i):
        self.phase, self.d, self.cnt = 0, 1.0, 0

    def act(self, obs, env):
        if self.phase == 0:
            a = np.array([self.d, 0.0], dtype=np.float32)
            if abs(env.pos_x) > 9.3:
                self.phase, self.cnt = 1, 0
        else:
            a = np.array([0.0, 1.0], dtype=np.float32)
            self.cnt += 1
            if self.cnt > 8:
                self.phase, self.d = 0, -self.d
        return a


class GreedyPolicy(_Base):
    """Velocity controller toward the nearest uncovered contour pixel (inputs: unit direction, own velocity)."""

    def __init__(self, speed=6.0):
        self.speed, self.name = float(speed), f"greedy_v{speed:g}"

    def act(self, obs, env):
        fx, fy = env.get_pixel_pos_f()
        x, y, _ = env._find_nearest_uncovered(fx, fy)
        return _thrust(env, x - fx, y - fy, self.speed)


class PurePursuitPolicy(_Base):
    """Pure pursuit on the ordered contour polylines (same Canny/findContours pipeline as BeadEnv).
    Pick the nearest polyline that is < 90% covered, jump to its nearest vertex, chase the vertex `lookahead`
    ahead (advancing as the bead gets within `tol` px); re-pick when the polyline reaches 90% covered or the
    whole loop has been traversed.  Nothing left to pick -> zero thrust."""

    def __init__(self, speed=6.0, lookahead=3, tol=6.0, skip_frac=0.9):
        self.speed, self.la, self.tol, self.skip = float(speed), lookahead, tol, skip_frac
        self.name = f"pure_pursuit_v{speed:g}"
        self._env_id = None

    def _prepare(self, env):
        gray = cv2.cvtColor(env.target, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cnts = [c[:, 0, :] for c in cnts if cv2.arcLength(c, True) > 30]
        S = env.IMG_SIZE
        table = -np.ones((S, S), dtype=np.int64)
        table[env._contour_pixels[:, 1], env._contour_pixels[:, 0]] = np.arange(len(env._contour_pixels))
        self.polys = [c.astype(np.float64) for c in cnts]
        self.pidx = [table[c[:, 1], c[:, 0]] for c in cnts]
        self._env_id = id(env)

    def reset(self, env, i):
        if self._env_id != id(env):
            self._prepare(env)
        self.cur, self.i, self.steps_on = None, 0, 0

    def _covfrac(self, env, k):
        ix = self.pidx[k]
        ix = ix[ix >= 0]
        return float(env._cov_mask[ix].mean()) if len(ix) else 1.0

    def act(self, obs, env):
        fx, fy = env.get_pixel_pos_f()
        if self.cur is not None and (self._covfrac(env, self.cur) >= self.skip
                                     or self.steps_on >= len(self.polys[self.cur])):
            self.cur = None
        if self.cur is None:
            best, bd, bj = None, 1e18, 0
            for k, P in enumerate(self.polys):
                if self._covfrac(env, k) >= self.skip:
                    continue
                d = np.hypot(P[:, 0] - fx, P[:, 1] - fy)
                j = int(d.argmin())
                if d[j] < bd:
                    best, bd, bj = k, d[j], j
            if best is None:
                return np.zeros(2, dtype=np.float32)
            self.cur, self.i, self.steps_on = best, bj, 0
        P = self.polys[self.cur]
        n = len(P)
        while math.hypot(P[self.i][0] - fx, P[self.i][1] - fy) < self.tol and self.steps_on < n:
            self.i = (self.i + 1) % n
            self.steps_on += 1
        tx, ty = P[(self.i + self.la) % n]
        return _thrust(env, tx - fx, ty - fy, self.speed)


class CropGreedyPolicy(_Base):
    """Uses ONLY the agent's own observation: the 64x64 distance-map crop (contour = 0, already-covered contour
    erased to 255, far = 255) and own velocity (state[2:4]).  No u-hat, no u_t, no global position.

    Rule: among crop pixels equal to 0 (remaining contour visible in the crop) steer to the one nearest the crop
    centre (= the bead), i.e. descend the visible distance map to the nearest remaining contour pixel, at
    `speed`.  If no remaining contour is visible, keep the current heading at `speed` (heading = last velocity
    direction; +x at the start).  Deliberately simple: it cannot plan across blank space and gets lost when the
    next contour piece is outside the 64 px window."""

    def __init__(self, speed=6.0):
        self.speed, self.name = float(speed), f"crop_greedy_v{speed:g}"

    def reset(self, env, i):
        self.heading = (1.0, 0.0)

    def act(self, obs, env):
        crop = obs["local_map"][..., 0]
        vx, vy = float(obs["state"][2]) * env.MAX_SPEED, float(obs["state"][3]) * env.MAX_SPEED
        ys, xs = np.nonzero(crop == 0)
        tx, ty = self.heading
        if len(xs):
            dx, dy = xs - 32.0, ys - 32.0
            j = int(np.argmin(dx * dx + dy * dy))
            if abs(dx[j]) + abs(dy[j]) > 0:      # sitting exactly on it: keep heading
                tx, ty = dx[j], dy[j]
        if math.hypot(vx, vy) > 0.5:
            self.heading = (vx, vy)
        return _thrust(env, tx, ty, self.speed, vx, vy)


def make_baseline(kind, speed=6.0):
    if kind in ("random", "zero", "raster"):
        return {"random": RandomPolicy, "zero": ZeroPolicy, "raster": RasterPolicy}[kind]()
    return {"greedy": GreedyPolicy, "pure_pursuit": PurePursuitPolicy,
            "crop_greedy": CropGreedyPolicy}[kind](speed)
