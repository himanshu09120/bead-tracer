# Bead Curve Reconstruction — Reinforcement Learning

Train a Stable-Baselines3 **PPO** agent to steer a physically simulated bead
along the contour extracted from a target image, maximizing genuine,
per-pixel contour coverage — without reward hacking.

```
Target Image → BeadEnv → Observation (image + state) → PPO Agent → Continuous 2-D Action → Bead Physics → Contour Coverage
```

---

## 1. Project objective

Given a target image, `BeadEnv` extracts its contour (Canny edges →
`findContours`) and simulates a bead with realistic 2-D physics (thrust,
drag, boundary bounce). The RL agent observes a local crop around the bead
plus a small numeric state vector, and outputs a continuous 2-D thrust
vector every physics tick. The objective is to trace as much of the target
contour as possible — measured as the fraction of **unique target-contour
pixels** the bead's path has swept over — while resisting reward-hacking
strategies (raster-sweeping across lines, camping on already-traced curve,
oscillating in place) that the environment's reward function is explicitly
designed to make unprofitable.

**The environment (`env.py`) is untouched.** Every file in this project is
built strictly against its existing public interface: constructor keyword
arguments, `reset()`, `step()`, `eval_start_points()`,
`reset(options={"fixed_start": ...})`, `render_coverage()`, and its own
built-in test suite (`_run_tests()`, i.e. `python env.py`).

## 2. Overall architecture

```
                     ┌───────────────────────────────────────────┐
                     │                  env.py                    │
                     │   BeadEnv (UNMODIFIED — provided as-is)     │
                     │   physics · reward · coverage · spaces      │
                     └───────────────────────────────────────────┘
                                   ▲              │
                        action (2,)│              │ obs: Dict{local_map, state}
                                   │              ▼
┌────────────┐   ┌───────────┐   ┌──────────────────────────────┐
│ config.py  │──▶│ train.py  │◀─▶│  SB3 PPO (MultiInputPolicy)   │
│ (all knobs)│   │           │   │  BeadFeaturesExtractor:        │
└────────────┘   └───────────┘   │   CNN(local_map)+MLP(state)    │
      │                          │   → fusion → pi/vf heads       │
      │          ┌────────────┐  └──────────────────────────────┘
      ├─────────▶│evaluate.py │  deterministic eval_start_points()
      │          └────────────┘  sweep, coverage-only metrics
      │          ┌─────────────┐
      └─────────▶│inference.py │  single-episode run + visualization
                 └─────────────┘
                 ┌────────────┐
                 │test_env.py │  env self-tests + RL smoke test
                 └────────────┘
```

`config.py` is the only place that defines PPO hyperparameters, the neural
network architecture, paths, and evaluation settings — nothing important is
scattered across the other scripts.

## 3. Environment

`BeadEnv` (Gymnasium `Env`) works on an 800×800 working raster derived from
the target image:

1. Canny edge detection + `findContours` extracts the target contour.
2. A distance-transform map (`distanceTransform`) gives distance-to-nearest-
   contour-pixel everywhere, used both for observations and reward shaping.
3. A bead (`ARENA` = 10×10 world units, radius 0.55) is simulated with
   thrust/drag/friction physics at `DT = 1/60s`, bouncing off the arena
   boundary.
4. Coverage is tracked as a **per-episode boolean mask over every individual
   target-contour pixel**, updated by a swept-capsule (point-to-segment)
   test each step — not by sampling the bead's position (which would tunnel
   through the curve at speed) and not by any coarser block/grid proxy.

Training uses the environment's own randomized curriculum spawn (a random
contour pixel + small jitter on every `reset()`). Evaluation and inference
use the environment's own deterministic support instead
(`eval_start_points(n)` + `reset(options={"fixed_start": (px, py)})`), so
evaluation is fully reproducible and never touches training's random resets.

## 4. Action space

`Box(-1.0, 1.0, shape=(2,), dtype=float32)` — a continuous 2-D thrust
vector, clipped to the unit disk. Unchanged. PPO's `MultiInputPolicy` uses a
`DiagGaussianDistribution` action head appropriate for continuous Box
spaces (this is SB3's default for continuous actions — nothing custom was
written here).

## 5. Observation space

`Dict`:

| Key | Shape | dtype | Content |
|---|---|---|---|
| `local_map` | (64, 64, 1) | uint8 | Local crop of the distance-transform map centered on the bead, with already-covered contour pixels erased (set to "far") so a memoryless policy can see what's left to trace. |
| `state` | (11,) | float32 | Position (2), velocity (2), unit direction to nearest uncovered pixel (2), normalized distance to nearest uncovered pixel (1), normalized distance to nearest contour pixel of any kind (1), coverage fraction rescaled to [-1,1] (1), previous action (2). |

SB3 handles this automatically and correctly for a Dict observation with an
image-like subspace: `local_map` is recognized as a channels-last image
space (`is_image_space`), so PPO's environment wrapping applies
`VecTransposeImage` (NHWC → NCHW) and per-step preprocessing divides pixel
values by 255 before any network sees them — no manual normalization code
was needed in this project.

## 6. Reward

Entirely defined in `env.py` (see the file's own v5 docstring for the full
history/rationale) and **not modified**. In summary, per step:

1. **Coverage term** — proportional to newly covered contour pixels this
   step (one-time per pixel, by construction of the mask).
2. **Continuity bonus** — small extra reward when new coverage extends an
   already-traced neighbourhood rather than an isolated hit.
3. **Approach shaping** — small, clipped reward for closing the distance to
   the nearest *uncovered* pixel, suppressed on steps that already earned
   real coverage.
4. **Action-smoothness penalty** — small, capped penalty for abrupt thrust
   changes.
5. **Far-from-contour penalty** — small, capped quadratic penalty for
   drifting far from the contour entirely.
6. **Anti-oscillation/anti-camping penalty** — charged only on steps that
   cover nothing new *and* either re-enter a recently-left grid cell or
   barely move at all.
7. **Completion bonus** (+100, one-time) — when true coverage crosses
   `completion_frac` (default 0.80), which also **terminates** the episode.

The environment's own test suite (§ "Testing" below) verifies structurally
that idle/off-contour/raster-sweep policies cannot out-earn honest tracing.

## 7. Coverage metric

**The single source of truth is `env.coverage_info()`** (`coverage`,
`coverage_pct`, `covered_pixels`, `total_contour_pixels`), read directly off
the environment's per-episode boolean mask over individual contour pixels.
No script in this project recomputes, approximates, or substitutes a
different notion of coverage (not trajectory length, not visited grid
blocks, not step count, not reward, not proximity events). `evaluate.py` and
`inference.py` report exactly these four values, verbatim from the
environment.

## 8. PPO

Stable-Baselines3's `PPO` implementation is used as-is (`stable_baselines3.PPO`)
— **no PPO logic is reimplemented**. `config.PPO_KWARGS` sets:

| Param | Value | Why |
|---|---|---|
| `policy` | `MultiInputPolicy` | required for a `Dict` observation space |
| `learning_rate` | 3e-4 | standard PPO default, stable for this problem size |
| `n_steps` | 1024 | per-env rollout length before each update |
| `batch_size` | 256 | minibatch size for the PPO epochs |
| `n_epochs` | 10 | PPO epochs per rollout |
| `gamma` | 0.995 | see § GAE below |
| `gae_lambda` | 0.95 | see § GAE below |
| `clip_range` | 0.2 | standard PPO clip |
| `ent_coef` | 0.005 | small exploration bonus (see below) |
| `vf_coef` | 0.5 | standard |
| `max_grad_norm` | 0.5 | standard |

**Why `ent_coef = 0.005` specifically for this environment:** the reward
function includes anti-oscillation/anti-camping and action-smoothness
penalties that a purely exploitative, undertrained policy could satisfy by
barely moving. A small entropy bonus keeps exploration alive long enough for
the agent to discover that actually tracing the contour pays far more than
sitting still (the environment's own tests confirm a zero-thrust policy
earns negative return).

## 9. GAE (Generalized Advantage Estimation)

Used exactly as SB3's `PPO` provides it — `gae_lambda` is a first-class PPO
constructor argument, and no custom advantage code exists anywhere in this
project.

- **`gamma = 0.995`** (effective horizon ≈ 200 steps): the environment pays
  dense per-step coverage reward (which favors a lower gamma, for lower
  value-target variance), but also a large, delayed completion bonus
  (+100) that can land many steps after the actions that enabled it, over
  episodes up to `max_steps = 2500` long. 0.995 values that delayed bonus
  properly without inflating variance the way something like 0.999 would.
- **`gae_lambda = 0.95`**: the standard, well-tested default for fixed-rate
  continuous-control physics tasks (this env steps at 60 Hz, structurally
  similar to MuJoCo-style control tasks where 0.95 is the established
  default). It trades a small amount of bias for a large reduction in
  advantage-estimate variance versus raw Monte-Carlo returns (`lambda=1`),
  which matters here because individual step rewards mix a dense coverage
  term with several smaller shaping/penalty terms.

## 10. Advantage estimation — termination vs. truncation

`BeadEnv.step()` correctly distinguishes `terminated` (task completion, a
real end-of-MDP event) from `truncated` (the neutral `max_steps`/stagnation
cutoff — explicitly commented in `env.py` as "neutral cutoff; the trainer
bootstraps V"). This project does not wrap the environment in anything that
could confuse the two (no extra `TimeLimit` wrapper is applied — the
environment manages its own step budget). Stable-Baselines3's
`OnPolicyAlgorithm.collect_rollouts()` handles this distinction correctly by
construction: on `truncated` (but not `terminated`) it bootstraps the value
function for the final observation before computing GAE advantages; on
`terminated` it does not. This is standard SB3 behavior for Gymnasium's
5-tuple `step()` API and required no additional code in this project.

## 11. Neural-network architecture

Defined in `config.py` as `BeadFeaturesExtractor` (a custom
`BaseFeaturesExtractor` subclass), used via `POLICY_KWARGS`:

```
local_map (1,64,64)                         state (11,)
        │                                        │
   Conv2d(1→16, k5 s2) + ReLU                Linear(11→64)+ReLU
   Conv2d(16→32,k3 s2) + ReLU                Linear(64→64)+ReLU
   Conv2d(32→32,k3 s2) + ReLU
        │  Flatten → Linear(→128) + ReLU         │
        └──────────────────┬─────────────────────┘
                    concat (128 + 64 = 192)
                            │
                  Linear(192 → 256) + ReLU   =  features_dim
                            │
              ┌─────────────┴─────────────┐
        pi: [128,128] MLP           vf: [128,128] MLP
        (Gaussian action mean)      (state value)
```

**Why not SB3's default `NatureCNN`:** SB3's `MultiInputPolicy` would
already route `local_map` through `NatureCNN` automatically (since a
(64,64,1) Box qualifies as an image space) — that would technically work,
but `NatureCNN` was sized for 84×84×4 stacked Atari frames. A single-channel
64×64 local distance-map crop is much simpler visual content (basically:
where is the nearby contour and how far), so a smaller, purpose-built 3-conv
CNN was used instead, fused with a small MLP branch for the 11-dim numeric
state. Total policy size: **≈430k parameters** — small enough to train many
timesteps per minute on a normal CPU, let alone a GPU.

## 12. Training (`train.py`)

```bash
python train.py --image target_images/path1.png --timesteps 1000000
```

What it does, in order:
1. Seeds NumPy/PyTorch/Python's `random` via `stable_baselines3.common.utils.set_random_seed`.
2. Validates the environment with `stable_baselines3.common.env_checker.check_env` (skippable with `--skip-validation`).
3. Builds an `n_envs`-way vectorized `BeadEnv` (`SubprocVecEnv` by default,
   `DummyVecEnv` with `--no-subproc`), each sub-env wrapped in SB3's
   `Monitor` (writes `logs/<run_name>/env_<i>.monitor.csv` — "record
   training statistics").
4. Builds `PPO("MultiInputPolicy", ...)` with `config.PPO_KWARGS` /
   `config.POLICY_KWARGS` (or resumes from `--resume <checkpoint>.zip`).
5. Trains with `model.learn(..., callback=[CheckpointCallback, CoverageLoggingCallback])`.
   - `CheckpointCallback` (SB3-provided, not reimplemented) periodically
     saves to `checkpoints/`, every `config.CHECKPOINT_FREQ_TIMESTEPS`
     timesteps.
   - `CoverageLoggingCallback` (this project's only custom callback) reads
     the environment's own `coverage_info()` off each completed episode's
     `info` dict and logs `rollout/ep_coverage_pct`,
     `rollout/ep_covered_pixels`, `rollout/ep_total_contour_pixels`,
     `rollout/ep_task_complete_rate`, `rollout/ep_steps` to TensorBoard —
     it does not compute or alter coverage, only reports it.
6. Saves the final model to `models/bead_ppo_final.zip` and a timestamped
   copy `models/<run_name>_final.zip`.

Key flags: `--image`, `--timesteps`, `--n-envs`, `--seed`, `--run-name`,
`--resume`, `--no-subproc`, `--device`, `--skip-validation`.

## 13. Evaluation (`evaluate.py`)

```bash
python evaluate.py --model models/bead_ppo_final.zip --image target_images/path1.png --n-points 20
```

Strictly separate from training. For each of `--n-points` deterministic,
evenly spaced points from `env.eval_start_points(n)`, it calls
`env.reset(options={"fixed_start": point})` and runs `model.predict(obs,
deterministic=True)` to completion. All reported numbers come straight from
`env.coverage_info()`:

- per-episode: `coverage_pct`, `covered_pixels`, `total_contour_pixels`, `reward`, `steps`, `task_complete`
- aggregate: mean/std/min/max coverage %, mean reward, mean steps, completion rate

Results are saved as JSON to `evaluations/eval_<tag>.json`; coverage
visualizations (`env.render_coverage()`) are saved to `results/eval_<tag>/`.
Pass `--stochastic` to sample actions instead of using the deterministic
policy mean.

## 14. Inference (`inference.py`)

```bash
python inference.py --model models/bead_ppo_final.zip --image target_images/path1.png
python inference.py --model models/bead_ppo_final.zip --image target_images/path1.png --start 475,71
```

Loads a trained model, creates the environment, runs the (by default
deterministic) policy for one episode, prints final coverage / reward /
steps / completion, and saves `env.render_coverage()` to
`results/inference_<timestamp>.png` (or `--output <path>`).

## 15. TensorBoard

```bash
tensorboard --logdir tensorboard/
```

Shows SB3's built-in scalars (`rollout/ep_rew_mean`, `rollout/ep_len_mean`,
`train/*` losses, `time/fps`, etc.) alongside this project's coverage
scalars added by `CoverageLoggingCallback`
(`rollout/ep_coverage_pct`, `rollout/ep_covered_pixels`,
`rollout/ep_total_contour_pixels`, `rollout/ep_task_complete_rate`,
`rollout/ep_steps`). Each `train.py` run gets its own subdirectory, named
`--run-name` (default `bead_ppo_<timestamp>`).

## 16. Folder structure

```
bead-tracer/
├── env.py            # BeadEnv — UNMODIFIED, exactly as provided
├── config.py          # every PPO/model/path/eval setting, in one place
├── train.py            # training entry point + env/model factories + callbacks
├── evaluate.py          # deterministic evaluation over eval_start_points()
├── inference.py          # single-episode run + visualization
├── test_env.py            # env's own tests + RL smoke test
├── requirements.txt
├── README.md
├── models/          # bead_ppo_final.zip and timestamped copies
├── checkpoints/     # periodic SB3 CheckpointCallback saves
├── logs/            # per-run Monitor CSVs (training statistics)
├── tensorboard/     # TensorBoard event files, one subdir per run
├── evaluations/     # evaluate.py JSON results
├── results/         # coverage visualizations (eval + inference)
└── target_images/
    └── path1.png    # placeholder synthetic target (see below)
```

## 17. Installation

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Verified in this project's own environment: Python 3.9, `stable-baselines3`
2.7.1, `gymnasium` 1.1.1, `torch` 2.8.0, `opencv-python-headless` 5.0.0.

## 18. Commands

```bash
# 1. Test everything first
python test_env.py

# 2. Train
python train.py --image target_images/path1.png --timesteps 1000000

# 3. Watch training live
tensorboard --logdir tensorboard/

# 4. Evaluate deterministically
python evaluate.py --model models/bead_ppo_final.zip --image target_images/path1.png

# 5. Run inference + save a visualization
python inference.py --model models/bead_ppo_final.zip --image target_images/path1.png

# Resume training from a checkpoint
python train.py --resume checkpoints/bead_ppo_500000_steps.zip --timesteps 500000
```

## 19. Expected outputs

- **`test_env.py`**: `10/10` environment self-test checks pass, `10/10` RL
  smoke-test checks pass (environment creation, `check_env`, PPO init,
  observation/action compatibility, a short training run, save, load,
  inference, evaluation) — this was verified during development of this
  project, not just assumed.
- **`train.py`**: TensorBoard scalars trending — `rollout/ep_rew_mean`
  increasing, `rollout/ep_coverage_pct` increasing over training, periodic
  `.zip` files appearing in `checkpoints/`.
- **`evaluate.py`**: a JSON file in `evaluations/` and PNGs in `results/`
  showing, per fixed start point, the target contour in grey, the actually
  covered contour in green, and the bead's trajectory in orange — a
  well-trained policy should show mean coverage climbing well above what an
  untrained policy achieves (a random/untrained policy typically covers a
  few percent per episode before stagnation-truncating; see the numbers
  printed by `python env.py`'s own tests for the honest-tracing-vs-random
  reward gap this reward function guarantees).
- **`inference.py`**: one coverage visualization PNG plus a printed summary.

## 20. Troubleshooting

- **`FileNotFoundError: Target image not found`** — pass a real path with
  `--image`, or place an image at `target_images/path1.png`. This project
  ships a synthetic placeholder contour there (a wavy closed "necklace"
  curve) so every script runs out of the box; replace it with your own
  target image for real use.
- **`check_env` warnings about the image space** — these are informational
  (SB3 warns rather than errors on some Dict/image edge cases); the project
  validates end to end via `test_env.py`, not just `check_env` alone.
- **Training is slow** — reduce `--n-envs`, reduce `config.PPO_KWARGS["n_steps"]`,
  or pass `--device cpu`/`--device cuda` explicitly. Each `BeadEnv` builds
  an 800×800 contour/distance-transform raster on construction, so very
  high `--n-envs` costs proportionally more memory.
- **`SubprocVecEnv` hangs or errors on some platforms** — pass
  `--no-subproc` to fall back to single-process `DummyVecEnv`.
- **Wanting the interactive Panda3D-style `env.render()`** — that method's
  lazy `from contour import BeadSimulation` import depends on a separate
  simulation module not included with this project (and out of scope per
  "do not modify the environment"). Every script here instead uses
  `env.render_coverage()`, which is fully self-contained in `env.py` and
  needs no extra dependency.
- **Resuming training changes total_timesteps accounting** — `--resume`
  loads the checkpoint's own step counter; pass `--timesteps` as the
  *additional* budget for this run (SB3's `reset_num_timesteps=False` on
  resume keeps the counter continuous, which is what the TensorBoard x-axis
  will reflect).
