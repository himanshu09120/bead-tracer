"""
env.py -- BeadEnv: Gymnasium environment for bead contour reconstruction.

Action space  : Box(-1, 1, shape=(2,))  -- continuous 2-D thrust vector (kept)
Observation   : Dict{local_map (64,64,1) uint8, state (11,) float32}  (kept)

===============================================================================
v5 -- COVERAGE METRIC REWRITE + REWARD-HACK CLOSURE
===============================================================================

PROBLEM REPORTED: coverage jumps to 15/40/60% then collapses to ~0%.

DIAGNOSIS (measured against the v4 code, not assumed):

  (a) Coverage was NOT contour coverage.  v4 reported
          len(self.visited) / self.total_target_blocks
      i.e. the fraction of 5x5 *blocks* that had been touched.  Measured on
      path1.png the contour has 8,500 pixels spread over 1,508 blocks holding
      1..12 pixels each (mean 5.64), so a block containing a single contour
      pixel counted exactly as much as a block containing twelve.  Worse,
      `_credit_neighborhood()` marked the whole 3x3 BLOCK neighbourhood (a
      15x15 px region) as visited whenever the bead came within ON_THRESH
      (8 px) of any unvisited pixel -- crediting up to 9 blocks (0.6% of the
      whole metric) in ONE step, including blocks whose contour pixels the
      bead never got near.  Measured bias: block-coverage read 8-13% higher
      than true pixel coverage on identical trajectories.

  (b) Reward hacking WAS possible.  A hand-written "raster sweep" policy that
      never traces anything -- drive right, nudge down, drive left, repeat --
      scored +36.9 return per episode versus +3.1 for a random policy, purely
      by crossing contour lines and collecting `_new_block_r = 3.0` x up to 9
      blocks per crossing.  Reward was paid for PROXIMITY EVENTS, not for
      traced arc length.  Secondary hole: the shaping potential (distance to
      nearest unvisited pixel) was silently reset UPWARD after every credit
      with no negative delta charged, so each credit re-opened a fresh
      "descent" that could be paid for again -- a slow but real pump.

  (c) The fluctuation had three independent causes:
        1. Coverage was a SINGLE-EPISODE sample, never averaged.  `visited`
           clears every reset, the spawn point is random, and episode length
           varies wildly because of stagnation truncation.  Measured with a
           FIXED (random) policy over 10 spawns: 4.38, 1.39, 0.66, 0.40, 1.46,
           1.06, 1.13, 0.00, 0.60, 1.26 % -- a 0->4.4% swing with no policy
           change at all.
        2. Crediting was chunky (up to 0.6% banked in one step), so the metric
           moves in visible jumps rather than smoothly.
        3. "Deterministic" evaluation was not deterministic in the STATE.
           `PPOTrainer.evaluate()` called `env.reset()` with no options, so
           every eval episode started from a fresh random contour pixel with
           random jitter.  The `fixed_start` option existed in reset() but
           nothing ever passed it.

FIXES IN THIS VERSION:

  1. COVERAGE IS NOW TRUE CONTOUR-PIXEL COVERAGE:
         coverage = unique covered contour pixels / total contour pixels * 100
     backed by a per-episode boolean mask over `self._contour_pixels`.  A
     pixel can only flip False->True once, so revisiting a region CANNOT
     raise coverage -- that property is structural, not a tuned constant.

  2. COVERAGE IS COMPUTED BY A SWEPT-CAPSULE TEST, not by sampling the bead's
     position: `_cover_along_segment()` marks every contour pixel whose
     point-to-SEGMENT distance from the previous bead position to the current
     one is <= COVER_RADIUS.  At MAX_SPEED the bead moves ~10 px/step, so a
     position-sampled test would tunnel straight through the curve and
     under-report; the segment test makes the covered set equal the true
     swept footprint of the path.

  3. TOLERANCE IS CONFIGURABLE: `cover_radius_px` (default 8.0 px).  For
     scale: the bead's radius is 0.55 world units = ~22 px at the 800x800
     working resolution, and top speed is ~10 px/step -- so 8 px is a
     conservative "physically touching the curve" tolerance that still keeps
     consecutive footprints overlapping.

  4. COVERAGE IS COMPUTED INDEPENDENTLY OF REWARD.  `_cover_along_segment()`
     is pure bookkeeping over the mask; `self.coverage` reads only the mask.
     The reward function *consumes* the number of newly covered pixels, but
     nothing about the metric depends on any reward constant, and the metric
     is reported through `info` on every step and reset.

  5. REWARD REBUILT AROUND REAL COVERAGE:
       - coverage term pays `_cov_r` per 1% of the contour NEWLY covered, so
         it is exactly proportional to genuine traced arc length and is
         one-time per pixel by construction.  The raster-sweep exploit dies
         because crossing a line perpendicularly covers ~1 line-width of new
         pixels instead of banking 9 whole blocks.
       - block/neighbourhood crediting is GONE.
       - shaping (approach the nearest UNCOVERED pixel) kept but reduced and
         clipped to +/-0.5 so it can never outweigh real coverage.
       - NEW anti-oscillation penalty: re-entering a `revisit_cell_px` cell
         that was already occupied within the last `revisit_window` steps,
         while covering nothing new that step, costs `revisit_pen`.  This is
         action-DEPENDENT (making progress avoids it entirely), unlike the
         old flat per-step stagnation tax that was removed in v3 for being a
         gradient-free noise floor.
       - completion bonus now triggers on true pixel coverage.
       - action-smoothness penalty discourages violent thrust switching.
       - approach shaping is suppressed on genuine coverage steps, keeping
         distance shaping subordinate to the real objective.
       - a small continuity bonus favors extending already-traced contour
         regions instead of isolated proximity hits.

  6. EVALUATION SUPPORT: `eval_start_points(n)` returns n evenly spaced,
     deterministic contour pixels; `reset(options={"fixed_start": (px,py)})`
     starts exactly there with NO jitter.  Training keeps random spawns +
     jitter, so exploration coverage and deterministic evaluation coverage
     are measured on genuinely different, non-overlapping protocols.

  7. VISUALIZATION: `render_coverage()` renders target contour (grey) +
     actually covered contour (green) + agent trajectory (orange) with
     start/end markers and the coverage figure printed on the image.

KEPT UNCHANGED ON PURPOSE: PPO-side everything, the continuous Box(-1,1,(2,))
action space, the physics constants, the drag law, the 64x64x1 local-map crop
and the 11-dim state vector (so the CNN/observation architecture in
modelclaude.py needs no change).

Earlier history (condensed): v2 removed the on-curve STREAK bonus (camping
exploit); v3 moved all seeking signal from a static distance map to distance-
to-nearest-UNVISITED so camping on finished curve stopped paying, and added
the global direction/distance features that make the observation Markovian
once the local crop is exhausted; v4 replaced Discrete(9) with the continuous
thrust vector and fixed a drag regression that had silently capped cruise
speed at 24% of MAX_SPEED; v4.1 removed a flat on-curve presence bonus that a
zero-thrust policy could farm indefinitely.
"""

import math
from collections import defaultdict

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class BeadEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    # Working resolution of the internal contour raster.
    IMG_SIZE = 800

    def __init__(
        self,
        target_image,
        curriculum: bool = True,
        *,
        # -- coverage / tolerance configuration (v5) --------------------------
        # Radius (in pixels of the 800x800 raster) within which a contour
        # pixel counts as covered by the bead's swept path.  Bead radius is
        # ~22 px and max travel is ~10 px/step, so 8.0 is a conservative
        # "touching the curve" tolerance that still keeps consecutive
        # footprints overlapping.  Raise it for a more forgiving task, lower
        # it to demand tighter tracing.
        cover_radius_px: float = 8.0,
        completion_frac: float = 0.80,
        max_steps: int = 2500,
        stag_limit: int = 350,
        # -- anti-oscillation configuration (v5) -------------------------------
        revisit_cell_px: float = 16.0,
        revisit_window: int = 60,
        revisit_pen: float = 0.05,
    ):
        super().__init__()

        # -- Physics constants (unchanged) -------------------------------------
        self.ARENA      = 10.0
        self.BEAD_R     = 0.55
        self.boundary   = self.ARENA - self.BEAD_R
        self.ACCEL      = 18.0
        self.MAX_SPEED  = 15.0
        self.FRICTION   = 5.0
        self.BOUNCE     = 0.45
        self.DT         = 1.0 / 60.0

        self.curriculum = curriculum

        # -- Spaces (unchanged) -------------------------------------------------
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.CROP_SIZE    = 64
        self.observation_space = spaces.Dict({
            "local_map": spaces.Box(
                low=0, high=255,
                shape=(self.CROP_SIZE, self.CROP_SIZE, 1),
                dtype=np.uint8,
            ),
            "state": spaces.Box(low=-1.0, high=1.0, shape=(11,), dtype=np.float32),
        })

        # -- Build contour + distance map from the target image -----------------
        if target_image.ndim == 2:
            target_image = cv2.cvtColor(target_image, cv2.COLOR_GRAY2BGR)
        S = self.IMG_SIZE
        self.target = cv2.resize(target_image, (S, S))

        gray    = cv2.cvtColor(self.target, cv2.COLOR_BGR2GRAY)
        blur    = cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = cv2.Canny(blur, 50, 150)
        cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cnts    = [c for c in cnts if cv2.arcLength(c, True) > 30]

        self.contour_mask = np.zeros((S, S), dtype=np.uint8)
        cv2.drawContours(self.contour_mask, cnts, -1, 255, 1)

        self.distance_map_raw = cv2.distanceTransform(
            255 - self.contour_mask, cv2.DIST_L2, 5
        ).astype(np.float32)

        # Normalised copy for the observation: clip at 100 px -> [0,255] uint8
        self.distance_map_norm = (
            np.clip(self.distance_map_raw / 100.0, 0.0, 1.0) * 255
        ).astype(np.uint8)

        # Pre-pad ONCE (the base map is static) so _get_obs() doesn't re-pad an
        # 800x800 array every step.  In padded space, pixel (px, py) sits at
        # (px + half, py + half).
        self._half = self.CROP_SIZE // 2
        self._padded_norm = np.pad(
            self.distance_map_norm, self._half,
            mode="constant", constant_values=255,
        )

        # -- Contour pixel table -- the ONLY basis for the coverage metric ------
        ys, xs = np.where(self.contour_mask > 0)
        self._contour_pixels = np.column_stack([xs, ys]).astype(np.int32)  # (N,2)
        self._cx = self._contour_pixels[:, 0].astype(np.float32)
        self._cy = self._contour_pixels[:, 1].astype(np.float32)
        self.total_contour_pixels = max(int(len(self._contour_pixels)), 1)

        # -- Block bookkeeping -- DIAGNOSTIC / OBSERVATION ONLY (not coverage) --
        # A block is "visited" only once EVERY contour pixel inside it is
        # covered.  Kept because `self.visited` is a convenient human-readable
        # progress readout and preserves the old attribute for any external
        # script; it is NEVER used to compute `self.coverage`.
        self.BLOCK = 5
        blocks_xy = np.column_stack([xs // self.BLOCK, ys // self.BLOCK])
        _b2i = defaultdict(list)
        for i, (gx, gy) in enumerate(map(tuple, blocks_xy.tolist())):
            _b2i[(gx, gy)].append(i)
        self._blocks             = list(_b2i.keys())
        self.target_blocks       = set(self._blocks)
        self.total_target_blocks = max(len(self._blocks), 1)
        block_index              = {b: k for k, b in enumerate(self._blocks)}
        self._pix_block_id       = np.empty(self.total_contour_pixels, dtype=np.int32)
        self._block_total        = np.zeros(len(self._blocks), dtype=np.int32)
        for b, idxs in _b2i.items():
            k = block_index[b]
            self._pix_block_id[np.asarray(idxs, dtype=np.int64)] = k
            self._block_total[k] = len(idxs)

        # -- Coverage / tolerance configuration ----------------------------------
        self.COVER_RADIUS   = float(cover_radius_px)
        self.completion_frac = float(completion_frac)
        self.FAR_THRESH     = 50.0     # px: "drifted off into blank space"
        self._NORM_DIST     = 100.0    # normalisation for distance features
        self.ERASE_R        = 2        # half-width of the obs-map erase square

        self.max_steps   = int(max_steps)
        self._stag_limit = int(stag_limit)

        # -- Anti-oscillation configuration --------------------------------------
        self._revisit_cell   = float(revisit_cell_px)
        self._revisit_window = int(revisit_window)
        self._revisit_pen    = float(revisit_pen)
        # Below this per-step displacement (px) the bead counts as camping.
        self._idle_dist_px   = 1.0

        # -- Reward weights -------------------------------------------------------
        # Coverage term: `_cov_r` reward per 1% of the contour newly covered.
        # Scale-free w.r.t. contour length: fully tracing the contour is worth
        # 100 * _cov_r regardless of image size.  Tracing at top speed covers
        # roughly 0.1-0.3% per step -> ~0.3-0.9 reward/step, i.e. the intended
        # "per-step reward lives in ~[-1, +1]" regime.
        self._cov_r        = 4.0
        self._dist_scale   = 0.015     # reduced: coverage remains the main objective
        self._dist_clip    = 0.3       # tighter cap so shaping cannot dominate coverage
        self._far_coef     = 0.0002    # quadratic far-from-any-contour penalty
        # ...capped: uncapped it reached ~12/step at 300 px from the curve and
        # ~40/step in the far corner of the arena, i.e. 40x outside the
        # intended "per-step reward in [-1,+1]" band, which would dominate
        # everything else the critic has to fit.  The cap is deliberately
        # SMALL because 38.7% of the arena is further than FAR_THRESH from the
        # contour (measured on path1.png): at -1.0/step that region would cost
        # more than the entire contour is worth to trace, and the agent would
        # learn to avoid crossing blank space even when the only remaining
        # uncovered segment is on the other side of it.  The symmetric
        # approach shaping already punishes drifting away; this is only a
        # safety net.
        self._far_max      = 0.15
        self._completion_r = 100.0

        # Small action-smoothness penalty.  This discourages violent
        # left-right / up-down action switching without competing with
        # genuine contour coverage.
        self._smooth_coef = 0.02
        self._smooth_max  = 0.05

        # Small continuity bonus.  Newly covered contour pixels that extend
        # an already covered neighbourhood are preferred over isolated
        # proximity hits.  It is deliberately much smaller than coverage.
        self._continuity_r = 0.15


        self.sim = None

        # Episode state (populated by reset()).
        self.visited  = set()
        self._cov_mask = np.zeros(self.total_contour_pixels, dtype=bool)
        self._covered  = 0
        self._last_continuous_new = 0
        self._last_new_px = 0
        self._traj     = []
        self.step_count = 0
        self.task_complete = False

    # =========================================================================
    # Coverage metric -- the single source of truth
    # =========================================================================

    @property
    def coverage(self) -> float:
        """Fraction in [0,1] of UNIQUE target-contour pixels covered this
        episode.  Derived exclusively from the per-episode boolean mask, so it
        is monotone non-decreasing, cannot be inflated by revisiting, and does
        not depend on any reward constant."""
        return self._covered / self.total_contour_pixels

    @property
    def coverage_pct(self) -> float:
        """Coverage as a percentage: covered / total * 100."""
        return 100.0 * self.coverage

    @property
    def covered_pixels(self) -> int:
        return int(self._covered)

    def coverage_info(self) -> dict:
        return {
            "coverage":             self.coverage,
            "coverage_pct":         self.coverage_pct,
            "covered_pixels":       int(self._covered),
            "total_contour_pixels": self.total_contour_pixels,
            "task_complete":        bool(self.task_complete),
            "steps":                int(self.step_count),
        }

    def _cover_along_segment(self, x0, y0, x1, y1) -> int:
        """Mark every not-yet-covered contour pixel whose distance to the
        LINE SEGMENT (x0,y0)->(x1,y1) is <= COVER_RADIUS.

        Returns the number of newly covered pixels.  Coverage remains a
        one-time per-pixel event.  In addition, records how many of those
        newly covered pixels were connected to already-covered contour
        pixels, which is used only for a small continuity bonus in reward.
        """
        r = self.COVER_RADIUS
        lo_x, hi_x = min(x0, x1) - r, max(x0, x1) + r
        lo_y, hi_y = min(y0, y1) - r, max(y0, y1) + r

        cand = (
            (~self._cov_mask)
            & (self._cx >= lo_x) & (self._cx <= hi_x)
            & (self._cy >= lo_y) & (self._cy <= hi_y)
        )
        idx = np.flatnonzero(cand)

        self._last_continuous_new = 0
        self._last_new_px = 0

        if idx.size == 0:
            return 0

        qx, qy = self._cx[idx], self._cy[idx]
        vx, vy = x1 - x0, y1 - y0
        seg_len2 = vx * vx + vy * vy
        if seg_len2 < 1e-12:
            t = np.zeros(idx.size, dtype=np.float32)
        else:
            t = np.clip(
                ((qx - x0) * vx + (qy - y0) * vy) / seg_len2,
                0.0, 1.0
            )

        ex = qx - (x0 + t * vx)
        ey = qy - (y0 + t * vy)
        hit = idx[(ex * ex + ey * ey) <= r * r]

        if hit.size == 0:
            return 0

        # Continuity is measured BEFORE updating the coverage mask.
        # A new pixel is considered continuous if a nearby contour pixel
        # (within ~sqrt(2) px) was already covered.  This rewards extending
        # a traced curve rather than isolated hits.
        old_mask = self._cov_mask
        hp = self._contour_pixels[hit]
        neighbor_continuous = 0
        radius = 2.0

        for qx_i, qy_i in hp:
            dx = self._cx - float(qx_i)
            dy = self._cy - float(qy_i)
            nearby = (dx * dx + dy * dy) <= radius * radius
            nearby[hit] = False
            if np.any(old_mask & nearby):
                neighbor_continuous += 1

        self._last_continuous_new = neighbor_continuous
        self._last_new_px = int(hit.size)

        self._cov_mask[hit] = True
        self._covered += int(hit.size)
        self._on_pixels_covered(hit)
        return int(hit.size)

    def _on_pixels_covered(self, hit: np.ndarray):
        """Bookkeeping that follows from newly covered pixels: erase them from
        the OBSERVED map (so a memoryless policy can see what is still left to
        trace) and update the fully-covered block set."""
        h    = self.ERASE_R
        half = self._half
        pxs  = self._contour_pixels[hit, 0] + half
        pys  = self._contour_pixels[hit, 1] + half
        for x, y in zip(pxs.tolist(), pys.tolist()):
            self._obs_map[y - h : y + h + 1, x - h : x + h + 1] = 255

        ids = self._pix_block_id[hit]
        np.subtract.at(self._block_remaining, ids, 1)
        uniq = np.unique(ids)
        done = uniq[(self._block_remaining[uniq] <= 0) & (~self._block_done[uniq])]
        for k in done.tolist():
            self._block_done[k] = True
            self.visited.add(self._blocks[k])

    def _find_nearest_uncovered(self, px, py):
        """Nearest not-yet-covered contour pixel to (px, py) -- brute-force
        vectorised numpy over the remaining pixels (a few thousand points; a
        KD-tree would be overkill and scipy isn't installed here).  Returns
        (x, y, dist_px); falls back to (px, py, 0.0) when everything is
        covered."""
        if self._covered >= self.total_contour_pixels:
            return float(px), float(py), 0.0
        m  = ~self._cov_mask
        dx = self._cx[m] - px
        dy = self._cy[m] - py
        d2 = dx * dx + dy * dy
        i  = int(np.argmin(d2))
        xs = self._cx[m]
        ys = self._cy[m]
        return float(xs[i]), float(ys[i]), float(math.sqrt(float(d2[i])))

    # =========================================================================
    # Gym API
    # =========================================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # -- Episode bookkeeping --------------------------------------------------
        self.visited.clear()
        self.step_count          = 0
        self.task_complete       = False
        self._last_coverage_step = 0
        self._prev_action        = np.zeros(2, dtype=np.float32)

        self._cov_mask   = np.zeros(self.total_contour_pixels, dtype=bool)
        self._covered    = 0
        self._last_continuous_new = 0
        self._last_new_px = 0
        self._block_remaining = self._block_total.copy()
        self._block_done      = np.zeros(len(self._blocks), dtype=bool)
        self._obs_map    = self._padded_norm.copy()
        self._traj       = []
        self._cell_seen  = {}

        # -- Starting position ------------------------------------------------------
        # options={"fixed_start": (px, py)} -> exact pixel, NO jitter.  Used by
        # deterministic evaluation so eval coverage is reproducible; training
        # keeps the random curriculum spawn + jitter.
        fixed_start = options.get("fixed_start") if options else None

        if fixed_start is not None:
            cpx = int(np.clip(int(fixed_start[0]), 0, self.IMG_SIZE - 1))
            cpy = int(np.clip(int(fixed_start[1]), 0, self.IMG_SIZE - 1))
        elif self.curriculum and len(self._contour_pixels) > 0:
            idx = int(self.np_random.integers(0, len(self._contour_pixels)))
            cpx, cpy = (int(v) for v in self._contour_pixels[idx])
        else:
            cpx, cpy = self.IMG_SIZE // 2, self.IMG_SIZE // 2

        span = float(self.IMG_SIZE - 1)
        self.pos_x = float(np.clip(float(cpx) / span * 20.0 - 10.0,
                                   -self.boundary, self.boundary))
        self.pos_y = float(np.clip(float(cpy) / span * 20.0 - 10.0,
                                   -self.boundary, self.boundary))

        if fixed_start is None:
            self.pos_x = float(np.clip(
                self.pos_x + self.np_random.uniform(-0.12, 0.12),
                -self.boundary, self.boundary))
            self.pos_y = float(np.clip(
                self.pos_y + self.np_random.uniform(-0.12, 0.12),
                -self.boundary, self.boundary))

        self.vel_x = 0.0
        self.vel_y = 0.0

        fx, fy = self.get_pixel_pos_f()
        self._prev_fx, self._prev_fy = fx, fy
        self._traj.append((fx, fy))
        self._cur_cell = (int(fx // self._revisit_cell),
                          int(fy // self._revisit_cell))
        self._cell_seen[self._cur_cell] = 0

        px, py = self.get_pixel_pos()
        self._last_dist_any = float(self.distance_map_raw[py, px])

        # Silently pre-cover the spawn footprint: the bead trivially occupies
        # its own spawn point, so counting that as covered is honest -- but
        # paying reward for it would be pure spawn luck, so no reward is given
        # and this happens before the first step() can see it.
        self._cover_along_segment(fx, fy, fx, fy)

        self._nn_x, self._nn_y, self.prev_uncovered_dist = (
            self._find_nearest_uncovered(fx, fy)
        )

        if self.sim is not None:
            self.sim.bead.setPos(self.pos_x, self.pos_y, self.BEAD_R)
            self.sim.vel_x = 0.0
            self.sim.vel_y = 0.0
            self.sim.tracer.clear()

        return self._get_obs(), self.coverage_info()

    def step(self, action):
        # -- Physics (unchanged from v4.2) -----------------------------------------
        ax, ay = float(action[0]), float(action[1])
        ax = float(np.clip(ax, -1.0, 1.0))
        ay = float(np.clip(ay, -1.0, 1.0))
        mag = math.hypot(ax, ay)
        if mag > 1.0:
            ax /= mag
            ay /= mag
            mag = 1.0

        self.vel_x += ax * self.ACCEL * self.DT
        self.vel_y += ay * self.ACCEL * self.DT

        # Drag scales with (1 - thrust magnitude): full thrust => zero drag
        # (reproduces the old discrete cruise speed), zero thrust => full drag
        # (reproduces the old braking), partial thrust interpolates.
        drag = max(0.0, 1.0 - mag)
        self.vel_x -= self.vel_x * self.FRICTION * self.DT * drag
        self.vel_y -= self.vel_y * self.FRICTION * self.DT * drag

        spd = math.hypot(self.vel_x, self.vel_y)
        if spd > self.MAX_SPEED:
            s = self.MAX_SPEED / spd
            self.vel_x *= s
            self.vel_y *= s

        self.pos_x += self.vel_x * self.DT
        self.pos_y += self.vel_y * self.DT

        if self.pos_x > self.boundary:
            self.pos_x = self.boundary
            self.vel_x = -abs(self.vel_x) * self.BOUNCE
        elif self.pos_x < -self.boundary:
            self.pos_x = -self.boundary
            self.vel_x = abs(self.vel_x) * self.BOUNCE

        if self.pos_y > self.boundary:
            self.pos_y = self.boundary
            self.vel_y = -abs(self.vel_y) * self.BOUNCE
        elif self.pos_y < -self.boundary:
            self.pos_y = -self.boundary
            self.vel_y = abs(self.vel_y) * self.BOUNCE

        # -- Coverage bookkeeping (computed BEFORE and INDEPENDENTLY of the
        #    reward -- the reward merely reads how many pixels were new) ----------
        fx, fy = self.get_pixel_pos_f()
        new_px = self._cover_along_segment(self._prev_fx, self._prev_fy, fx, fy)
        moved  = math.hypot(fx - self._prev_fx, fy - self._prev_fy)
        self._prev_fx, self._prev_fy = fx, fy
        self._traj.append((fx, fy))

        px, py = self.get_pixel_pos()
        curr_dist_any = float(self.distance_map_raw[py, px])
        nn_x, nn_y, curr_dist_uncov = self._find_nearest_uncovered(fx, fy)

        reward     = 0.0
        terminated = False
        truncated  = False

        # 1. COVERAGE REWARD -- strictly proportional to newly covered contour
        #    pixels, one-time per pixel by construction.
        if new_px:
            reward += self._cov_r * (100.0 * new_px / self.total_contour_pixels)
            self._last_coverage_step = self.step_count

            # Small continuity bonus: reward extending an already traced
            # neighbourhood, not isolated proximity hits.
            if self._last_continuous_new:
                continuity_frac = self._last_continuous_new / new_px
                reward += self._continuity_r * continuity_frac

        # 2. APPROACH SHAPING -- used mainly to find the remaining contour.
        #    Once genuine new coverage happens, coverage itself is the useful
        #    learning signal, so distance shaping is suppressed on that step.
        if new_px == 0:
            delta = self.prev_uncovered_dist - curr_dist_uncov
            reward += float(np.clip(
                self._dist_scale * delta,
                -self._dist_clip, self._dist_clip
            ))

        if new_px:
            self._nn_x, self._nn_y, self.prev_uncovered_dist = (
                self._find_nearest_uncovered(fx, fy)
            )
        else:
            self._nn_x, self._nn_y = nn_x, nn_y
            self.prev_uncovered_dist = curr_dist_uncov
        self._last_dist_any = curr_dist_any

        # 3. ACTION SMOOTHNESS -- small penalty for abrupt thrust changes.
        #    This is intentionally capped so it cannot outweigh coverage.
        action_arr = np.array([ax, ay], dtype=np.float32)
        action_delta = action_arr - self._prev_action
        smooth_penalty = min(
            self._smooth_coef * float(np.dot(action_delta, action_delta)),
            self._smooth_max
        )
        reward -= smooth_penalty

        # 4. Far-from-any-contour penalty (safety net against drifting into
        #    blank space).  Uses distance to ANY contour so it never fights a
        #    legitimate long transit toward a genuinely distant segment.
        if curr_dist_any > self.FAR_THRESH:
            excess = curr_dist_any - self.FAR_THRESH
            reward -= min(self._far_coef * excess * excess, self._far_max)

        # 5. ANTI-OSCILLATION / ANTI-CAMPING.  Costs `revisit_pen` on a step
        #    that covers nothing new AND either
        #      (a) RE-ENTERS a cell the bead had left within the last
        #          `revisit_window` steps -- i.e. genuine backtracking or
        #          oscillation (see _mark_cell for why re-entry, not mere
        #          occupancy, is the right trigger), or
        #      (b) barely moved at all -- camping.
        #    Both are action-dependent: covering new curve, or simply moving
        #    on to fresh ground, avoids the term entirely.  That is what
        #    distinguishes it from the flat per-step stagnation tax removed in
        #    v3, which applied identically no matter what the agent did and so
        #    carried no gradient at all.
        revisited = self._mark_cell(fx, fy)
        camping   = moved < self._idle_dist_px
        if new_px == 0 and (revisited or camping):
            reward -= self._revisit_pen

        # 6. Completion -- on TRUE pixel coverage.
        if not self.task_complete and self.coverage >= self.completion_frac:
            reward += self._completion_r
            self.task_complete = True
            terminated = True

        self._prev_action = action_arr
        self.step_count  += 1

        if not terminated:
            stale = self.step_count - self._last_coverage_step
            if self.step_count >= self.max_steps or stale >= self._stag_limit:
                truncated = True   # neutral cutoff; the trainer bootstraps V

        return self._get_obs(), reward, terminated, truncated, self.coverage_info()

    # =========================================================================
    # Helpers
    # =========================================================================

    def get_pixel_pos_f(self):
        """Bead position in (sub-pixel) raster coordinates."""
        span = float(self.IMG_SIZE - 1)
        return ((self.pos_x + 10.0) / 20.0 * span,
                (self.pos_y + 10.0) / 20.0 * span)

    def get_pixel_pos(self):
        fx, fy = self.get_pixel_pos_f()
        lim = self.IMG_SIZE - 1
        return int(np.clip(int(fx), 0, lim)), int(np.clip(int(fy), 0, lim))

    def _mark_cell(self, fx, fy) -> bool:
        """Record occupancy of the coarse cell containing (fx, fy).  Returns
        True only for a genuine RE-ENTRY: the bead left a cell and came back
        to it within `_revisit_window` steps.

        Testing continued occupancy instead would fire on any slow forward
        motion -- at 16 px cells and up to ~10 px of travel per step, a bead
        simply moving carefully along the curve stays in the same cell for
        several steps in a row, and an early random policy was collecting the
        penalty on literally every step (a flat -17.5 per 350-step episode).
        That is exactly the action-independent noise floor that had to be
        removed from earlier versions; keying on re-entry keeps the term
        pointed at real backtracking and oscillation."""
        key = (int(fx // self._revisit_cell), int(fy // self._revisit_cell))
        if key == self._cur_cell:
            return False                      # still in the same cell: not a revisit
        last = self._cell_seen.get(key)
        self._cell_seen[self._cur_cell] = self.step_count   # stamp the cell we left
        self._cur_cell = key
        self._cell_seen[key] = self.step_count
        return last is not None and (self.step_count - last) <= self._revisit_window

    def eval_start_points(self, n: int):
        """`n` deterministic, evenly spaced contour pixels -- pass one as
        reset(options={"fixed_start": pt}) so deterministic evaluation always
        replays the SAME starting states instead of random curriculum spawns."""
        if len(self._contour_pixels) == 0:
            return [(self.IMG_SIZE // 2, self.IMG_SIZE // 2)] * n
        idxs = np.linspace(0, len(self._contour_pixels) - 1, n).astype(np.int64)
        return [(int(self._contour_pixels[i, 0]), int(self._contour_pixels[i, 1]))
                for i in idxs]

    def _get_obs(self):
        px, py = self.get_pixel_pos()

        # In padded space the bead pixel maps to (px+half, py+half), so this
        # slice is centred on the bead.  _obs_map has covered curve erased.
        crop = self._obs_map[py : py + self.CROP_SIZE, px : px + self.CROP_SIZE]
        crop = np.expand_dims(crop, axis=-1)  # (64, 64, 1)

        ddx, ddy = self._nn_x - px, self._nn_y - py
        dnorm    = math.hypot(ddx, ddy)
        if dnorm > 1e-6:
            dir_x, dir_y = ddx / dnorm, ddy / dnorm
        else:
            dir_x, dir_y = 0.0, 0.0

        nn_dist_norm  = float(np.clip(self.prev_uncovered_dist / self._NORM_DIST, 0.0, 1.0))
        any_dist_norm = float(np.clip(self._last_dist_any / self._NORM_DIST, 0.0, 1.0))

        state = np.array([
            np.clip(self.pos_x / self.ARENA,     -1.0, 1.0),
            np.clip(self.pos_y / self.ARENA,     -1.0, 1.0),
            np.clip(self.vel_x / self.MAX_SPEED, -1.0, 1.0),
            np.clip(self.vel_y / self.MAX_SPEED, -1.0, 1.0),
            dir_x,
            dir_y,
            nn_dist_norm,
            any_dist_norm,
            self.coverage * 2.0 - 1.0,
            float(self._prev_action[0]),
            float(self._prev_action[1]),
        ], dtype=np.float32)

        return {"local_map": crop, "state": state}

    # =========================================================================
    # Visualization
    # =========================================================================

    def render_coverage(self, save_path: str = None, title: str = None):
        """Render target contour + ACTUALLY covered contour + agent trajectory.

          grey   = target contour still uncovered
          green  = target contour covered this episode (the coverage metric,
                   drawn straight from the same boolean mask it is computed
                   from -- so the picture cannot disagree with the number)
          orange = agent trajectory
          cyan dot = start, red dot = end
        """
        S = self.IMG_SIZE
        canvas = np.full((S, S, 3), 20, dtype=np.uint8)

        m_all = np.zeros((S, S), dtype=np.uint8)
        m_cov = np.zeros((S, S), dtype=np.uint8)
        m_all[self._contour_pixels[:, 1], self._contour_pixels[:, 0]] = 255
        if self._covered:
            cp = self._contour_pixels[self._cov_mask]
            m_cov[cp[:, 1], cp[:, 0]] = 255

        k = np.ones((3, 3), np.uint8)
        m_all = cv2.dilate(m_all, k)
        m_cov = cv2.dilate(m_cov, k)

        canvas[m_all > 0] = (105, 105, 105)     # uncovered target contour
        canvas[m_cov > 0] = (60, 235, 60)       # covered contour

        if len(self._traj) > 1:
            pts = np.asarray(self._traj, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], False, (0, 165, 255), 1, cv2.LINE_AA)
            cv2.circle(canvas, tuple(int(v) for v in self._traj[0]),  6, (255, 230, 0), -1)
            cv2.circle(canvas, tuple(int(v) for v in self._traj[-1]), 6, (0, 0, 255), -1)

        head = title or "BeadEnv coverage"
        lines = [
            head,
            f"coverage: {self.coverage_pct:.2f}%  "
            f"({self._covered:,}/{self.total_contour_pixels:,} contour px)",
            f"steps: {self.step_count}   tolerance: {self.COVER_RADIUS:.1f}px"
            f"   complete: {self.task_complete}",
        ]
        for i, txt in enumerate(lines):
            cv2.putText(canvas, txt, (12, 26 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)

        legend = [("target contour", (105, 105, 105)),
                  ("covered contour", (60, 235, 60)),
                  ("trajectory", (0, 165, 255))]
        for i, (txt, col) in enumerate(legend):
            y = S - 70 + 22 * i
            cv2.rectangle(canvas, (12, y - 10), (30, y + 4), col, -1)
            cv2.putText(canvas, txt, (38, y + 3), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (235, 235, 235), 1, cv2.LINE_AA)

        if save_path:
            cv2.imwrite(save_path, canvas)
        return canvas

    def render(self):
        if self.sim is None:
            from contour import BeadSimulation
            self.sim = BeadSimulation()
            self.sim.taskMgr.remove("update")
            self.sim.bead.setPos(self.pos_x, self.pos_y, self.BEAD_R)

        self.sim.bead.setPos(self.pos_x, self.pos_y, self.BEAD_R)
        self.sim.tracer.add_point(self.sim.bead.getPos())
        self.sim.graphicsEngine.renderFrame()

        frame = self.sim.capture_frame()
        cv2.imshow("BeadEnv", frame)
        cv2.waitKey(1)
        return frame

    def close(self):
        # Only meaningful if render() actually opened a window (self.sim is
        # set there). Guarded because opencv-python-headless -- the OpenCV
        # build this project's requirements.txt installs for training/eval,
        # which have no GUI use for render() -- has no highgui support and
        # raises cv2.error on destroyAllWindows() even when no window exists.
        if self.sim is not None:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


# ===============================================================================
# Self-tests for the coverage metric
#   python env.py            -> run the test suite
# ===============================================================================

def _make_env(**kw):
    """Synthetic target: a circle, so the ground truth is exactly known -- the
    fraction of the circle's angular sweep that was traced IS the fraction of
    contour pixels that should be covered (contour pixels are distributed
    uniformly in angle)."""
    img = np.zeros((400, 400), dtype=np.uint8)
    cv2.circle(img, (200, 200), 130, 255, 2)
    defaults = dict(curriculum=True, completion_frac=2.0,   # never auto-finish
                    max_steps=200_000, stag_limit=200_000)  # never auto-truncate
    defaults.update(kw)
    return BeadEnv(target_image=img, **defaults)


# Geometry of the synthetic target in the 800x800 working raster: the 400x400
# source is resized 2x, so the circle is centred at (400, 400) with radius 260.
_CX, _CY, _R = 400.0, 400.0, 260.0


def _arc_waypoints(f0: float, f1: float, spacing_px: float = 4.0):
    """Waypoints along the fraction [f0, f1] of the circle's circumference."""
    arc_len = 2.0 * math.pi * _R * abs(f1 - f0)
    n = max(int(arc_len / spacing_px), 2)
    ts = np.linspace(2.0 * math.pi * f0, 2.0 * math.pi * f1, n)
    return np.column_stack([_CX + _R * np.cos(ts), _CY + _R * np.sin(ts)])


def _drive_along(env, waypoints, speed=6.0, max_steps=100_000, tol=6.0,
                 lookahead=3):
    """Pure-pursuit controller: chase `waypoints` through the SAME env.step()
    physics the agent uses -- no teleporting and no direct writes to the
    coverage mask, so these tests exercise the real coverage code path.
    `speed` is a desired cruise speed in world units/s.
    Returns (steps_used, total_reward, ended_early)."""
    i, used, total_r = 0, 0, 0.0
    n = len(waypoints)
    while i < n and used < max_steps:
        fx, fy = env.get_pixel_pos_f()
        while i < n and math.hypot(waypoints[i][0] - fx,
                                   waypoints[i][1] - fy) < tol:
            i += 1
        if i >= n:
            break
        tx, ty = waypoints[min(i + lookahead, n - 1)]
        dx, dy = tx - fx, ty - fy
        d = math.hypot(dx, dy) or 1.0
        # Desired velocity (world units/s) toward the look-ahead waypoint.
        # Pixel and world axes are aligned (px = (pos+10)/20*799), so the
        # pixel-space direction is also the world-space direction.
        v_des_x, v_des_y = dx / d * speed, dy / d * speed
        # Thrust that would realise that velocity in one tick, clamped to the
        # unit disk exactly like the env does.
        a_x = (v_des_x - env.vel_x) / (env.ACCEL * env.DT)
        a_y = (v_des_y - env.vel_y) / (env.ACCEL * env.DT)
        m = math.hypot(a_x, a_y)
        if m > 1.0:
            a_x, a_y = a_x / m, a_y / m
        _, r, term, trunc, _ = env.step(np.array([a_x, a_y], dtype=np.float32))
        total_r += r
        used += 1
        if term or trunc:
            return used, total_r, True
    return used, total_r, False


def _run_tests() -> bool:
    results = []

    def check(name, ok, detail):
        results.append((name, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n         {detail}")

    print("=" * 78)
    print("  BeadEnv COVERAGE METRIC TESTS")
    print("=" * 78)

    # -- 1a. unit level: re-covering the same ground yields ZERO new pixels ----
    env = _make_env()
    env.reset(seed=0, options={"fixed_start": (int(_CX + _R), int(_CY))})
    a = (_CX + _R, _CY)
    b = (_CX + _R * math.cos(0.4), _CY + _R * math.sin(0.4))
    first  = env._cover_along_segment(a[0], a[1], b[0], b[1])
    cov_a  = env.coverage_pct
    second = env._cover_along_segment(a[0], a[1], b[0], b[1])
    third  = env._cover_along_segment(b[0], b[1], a[0], a[1])   # reverse
    cov_b  = env.coverage_pct
    check("re-covering identical ground yields 0 new pixels",
          first > 0 and second == 0 and third == 0 and cov_a == cov_b,
          f"1st pass over the segment: +{first} px ({cov_a:.2f}%); "
          f"2nd pass: +{second} px; reverse pass: +{third} px "
          f"-> coverage still {cov_b:.2f}%")

    # -- 1b. behavioural: re-tracing a traced arc doesn't raise coverage -------
    env = _make_env()
    arc = _arc_waypoints(0.0, 0.25)
    env.reset(seed=0, options={"fixed_start": (int(arc[0][0]), int(arc[0][1]))})
    _drive_along(env, arc)
    # Brake to a standstill BEFORE the measurement starts.  The bead carries
    # momentum out of the forward pass and needs ~8 deg of arc to reverse, so
    # measuring immediately would charge the coasting overshoot (which covers
    # genuinely fresh curve, correctly) against the revisit test.
    for _ in range(60):
        env.step(np.zeros(2, dtype=np.float32))
    cov_first = env.coverage_pct

    # Re-trace an INNER sub-arc (3%..22% of the circle) of the 0..25% arc just
    # covered.  The margin matters: the pursuit controller needs ~15-20 px to
    # reverse direction, so laps that ran to the very end of the traced arc
    # would coast past it onto genuinely fresh curve -- coverage would then
    # rise for a real reason, and the test would be measuring controller
    # overshoot rather than the metric.
    inner = _arc_waypoints(0.03, 0.22)
    revisit_steps, revisit_reward = 0, 0.0
    for lap in range(4):                     # 4 more laps over COVERED ground
        s, r, _ = _drive_along(env, inner[::-1] if lap % 2 == 0 else inner)
        revisit_steps  += s
        revisit_reward += r
    cov_repeat = env.coverage_pct
    check("repeated visits don't increase coverage",
          (cov_repeat - cov_first) < 0.5 and revisit_reward <= 0.0,
          f"{cov_first:.2f}% -> {cov_repeat:.2f}% "
          f"(+{cov_repeat - cov_first:.2f} pp over {revisit_steps} revisit "
          f"steps); reward earned while revisiting = {revisit_reward:+.2f} "
          f"(must be <= 0)")
    env.render_coverage("coverage_test_revisit.png",
                        title="TEST 1: quarter-arc traced, then re-traced 4x")

    # -- 2. no contour contact -> exactly 0% ------------------------------------
    env2 = _make_env()
    env2.reset(seed=1, options={"fixed_start": (8, 8)})     # far corner
    cov_reset = env2.coverage_pct
    tot_r = 0.0
    for t in range(300):                                    # jiggle in place
        act = np.array([math.cos(t * 0.7) * 0.5, math.sin(t * 0.7) * 0.5],
                       dtype=np.float32)
        _, r, term, trunc, _ = env2.step(act)
        tot_r += r
        if term or trunc:
            break
    check("no contour contact gives 0% coverage",
          cov_reset == 0.0 and env2.coverage_pct == 0.0,
          f"coverage at reset = {cov_reset:.2f}%, after {env2.step_count} "
          f"off-contour steps = {env2.coverage_pct:.2f}%")
    check("an off-contour policy earns no positive return",
          tot_r <= 0.0, f"return = {tot_r:+.2f}")

    # -- 3. partial traversal -> PROPORTIONAL coverage --------------------------
    #     Contour pixels are uniform in angle, so tracing a fraction f of the
    #     circumference must yield ~f coverage.  A small positive bias is
    #     expected: the COVER_RADIUS tolerance adds a half-disc of curve at
    #     each end of the traced arc (~2*8px of a 1,634px circumference ~ 1pp).
    partials = []
    for f in (0.25, 0.50, 0.75):
        e = _make_env()
        wp = _arc_waypoints(0.0, f)
        e.reset(seed=2, options={"fixed_start": (int(wp[0][0]), int(wp[0][1]))})
        _drive_along(e, wp)
        partials.append((f, e.coverage_pct, e))
    ok_partial = all(abs(cov - f * 100.0) <= 6.0 for f, cov, _ in partials)
    check("partial traversal gives proportional coverage",
          ok_partial,
          "  ".join(f"traced {f*100:.0f}% -> coverage {cov:.2f}%"
                    for f, cov, _ in partials)
          + "   (tolerance +/-6 pp)")
    partials[1][2].render_coverage(
        "coverage_test_partial.png", title="TEST 3: 50% of the contour traced")

    # -- 4. full traversal -> ~100% ----------------------------------------------
    env4 = _make_env()
    wp = _arc_waypoints(0.0, 1.0)
    env4.reset(seed=3, options={"fixed_start": (int(wp[0][0]), int(wp[0][1]))})
    _drive_along(env4, wp)
    cov_full = env4.coverage_pct
    check("full traversal approaches 100%",
          cov_full >= 95.0,
          f"coverage = {cov_full:.2f}% "
          f"({env4.covered_pixels:,}/{env4.total_contour_pixels:,} px)")
    env4.render_coverage("coverage_test_full.png",
                         title="TEST 4: full contour traced")

    check("partial coverage is strictly below full coverage",
          partials[1][1] < cov_full - 5.0,
          f"50% traversal {partials[1][1]:.2f}% vs full {cov_full:.2f}%")

    # -- 5. anti-hack: the raster sweep must not out-earn honest tracing --------
    #     On the OLD (v4) block reward this exploit scored +36.9 per episode
    #     while never tracing anything, versus +3.1 for a random policy.
    def raster(e, budget):
        e.reset(seed=4)
        R, d, phase, cnt = 0.0, 1.0, 0, 0
        for _ in range(budget):
            if phase == 0:
                act = np.array([d, 0.0], dtype=np.float32)
                if abs(e.pos_x) > 9.3:
                    phase, cnt = 1, 0
            else:
                act = np.array([0.0, 1.0], dtype=np.float32)
                cnt += 1
                if cnt > 8:
                    phase, d = 0, -d
            _, r, term, trunc, _ = e.step(act)
            R += r
            if term or trunc:
                break
        return R

    BUDGET = 400
    env5 = _make_env(stag_limit=350, max_steps=2500)
    R_raster   = raster(env5, BUDGET)
    cov_raster = env5.coverage_pct

    env6 = _make_env(stag_limit=350, max_steps=2500)
    wp6 = _arc_waypoints(0.0, 1.0)
    env6.reset(seed=4, options={"fixed_start": (int(wp6[0][0]), int(wp6[0][1]))})
    _, R_trace, _ = _drive_along(env6, wp6, max_steps=BUDGET)
    check("honest tracing out-earns the raster-sweep exploit",
          R_trace > R_raster,
          f"trace {R_trace:+.2f} ({env6.coverage_pct:.2f}% cov) vs "
          f"raster {R_raster:+.2f} ({cov_raster:.2f}% cov), "
          f"equal {BUDGET}-step budget")

    # -- 6. a do-nothing policy must not earn anything ---------------------------
    env7 = _make_env(stag_limit=350, max_steps=2500)
    env7.reset(seed=5)
    R_idle = 0.0
    for _ in range(400):
        _, r, term, trunc, _ = env7.step(np.zeros(2, dtype=np.float32))
        R_idle += r
        if term or trunc:
            break
    check("a zero-thrust policy earns no positive return",
          R_idle <= 0.0,
          f"return = {R_idle:+.2f} over {env7.step_count} idle steps "
          f"(coverage {env7.coverage_pct:.2f}%, all of it the free, "
          f"unpaid spawn footprint)")

    # -- 7. coverage is monotone non-decreasing within an episode ---------------
    env8 = _make_env(stag_limit=350, max_steps=2500)
    env8.reset(seed=6)
    prev, mono = env8.coverage, True
    rng = np.random.default_rng(0)
    for _ in range(400):
        _, _, term, trunc, info = env8.step(
            rng.uniform(-1, 1, size=2).astype(np.float32))
        mono &= (info["coverage"] >= prev - 1e-12)
        prev = info["coverage"]
        if term or trunc:
            break
    check("coverage is monotone non-decreasing within an episode",
          mono, f"{env8.step_count} random steps, final {prev*100:.2f}%")

    print("=" * 78)
    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"  {n_pass}/{len(results)} checks passed")
    print("  wrote coverage_test_revisit.png, coverage_test_partial.png, "
          "coverage_test_full.png")
    print("=" * 78)
    return n_pass == len(results)


if __name__ == "__main__":
    raise SystemExit(0 if _run_tests() else 1)
