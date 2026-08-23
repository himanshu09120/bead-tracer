"""
config.py -- Single source of truth for every project-level setting.

Nothing about the environment (env.py) is configured here beyond the kwargs
its own constructor already exposes (cover_radius_px, completion_frac,
max_steps, stag_limit, revisit_cell_px, revisit_window, revisit_pen,
curriculum). Those are BeadEnv's OWN documented configuration surface, not a
modification of it -- passing kwargs to an unmodified constructor is exactly
how the environment is meant to be tuned from the outside.

Everything else -- paths, PPO hyperparameters, the policy's neural network
architecture, evaluation/checkpoint settings, and the seed -- lives here so
no important number is buried inside train.py / evaluate.py / inference.py.
"""

import os

import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR      = os.path.join(PROJECT_ROOT, "models")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
LOGS_DIR        = os.path.join(PROJECT_ROOT, "logs")
TENSORBOARD_DIR = os.path.join(PROJECT_ROOT, "tensorboard")
EVALUATIONS_DIR = os.path.join(PROJECT_ROOT, "evaluations")
RESULTS_DIR     = os.path.join(PROJECT_ROOT, "results")
TARGET_IMAGES_DIR = os.path.join(PROJECT_ROOT, "target_images")

DEFAULT_IMAGE_PATH = os.path.join(TARGET_IMAGES_DIR, "path1.png")

FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "bead_ppo_final.zip")

for _d in (MODELS_DIR, CHECKPOINTS_DIR, LOGS_DIR, TENSORBOARD_DIR,
           EVALUATIONS_DIR, RESULTS_DIR, TARGET_IMAGES_DIR):
    os.makedirs(_d, exist_ok=True)

# =============================================================================
# Reproducibility
# =============================================================================

SEED = 42

# =============================================================================
# Environment configuration
# =============================================================================
# These are passed straight into BeadEnv(**ENV_KWARGS) -- they use the
# environment's own constructor keyword arguments, unchanged. Defaults here
# match the environment's own defaults; they are surfaced in one place so a
# reviewer doesn't have to go hunting through multiple scripts to see how the
# task is configured.

ENV_KWARGS = dict(
    cover_radius_px=8.0,
    completion_frac=0.95,
    max_steps=2500,
    stag_limit=350,
    revisit_cell_px=16.0,
    revisit_window=60,
    revisit_pen=0.05,
)

# Number of parallel environments used for training rollouts. Each is a full
# BeadEnv (800x800 contour raster + distance transform), so this is kept
# modest for a "normal machine" -- raise it if more CPU cores are available.
N_ENVS = 4

# Use SubprocVecEnv (separate OS processes) when N_ENVS > 1 so environment
# stepping is genuinely parallel; DummyVecEnv (single process) otherwise,
# which avoids multiprocessing overhead/pickling for N_ENVS == 1 and is also
# what test_env.py's smoke test uses for a fast, deterministic run.
USE_SUBPROC = True

# =============================================================================
# PPO hyperparameters
# =============================================================================
# Rationale (see README.md "PPO" and "GAE" sections for the full writeup):
#
#   gamma = 0.995
#     The environment pays dense per-step coverage reward (good for a low
#     gamma), but the completion bonus (+100, `_completion_r` in env.py) can
#     land many steps after the actions that made it possible, and episodes
#     run up to `max_steps` = 2500 steps. An effective horizon of
#     1/(1-gamma) = 200 steps balances valuing that delayed bonus against
#     keeping value-target variance manageable -- pure 0.99 (100-step
#     horizon) undervalues completion, and something like 0.999 (1000-step
#     horizon) makes the value function needlessly high variance given how
#     dense the per-step reward already is.
#
#   gae_lambda = 0.95
#     Standard, well-tested default for continuous-control tasks stepped at
#     a fixed physics rate (this env: DT = 1/60s, akin to MuJoCo-style
#     tasks where 0.95 is the well-established default). It trades a small
#     amount of bias for a large reduction in advantage-estimate variance
#     relative to lambda = 1 (raw Monte-Carlo returns), which matters here
#     because individual step rewards mix a dense coverage term with sparser
#     shaping/penalty terms.
#
#   n_steps / batch_size / n_epochs
#     n_steps=1024 per env x N_ENVS gives a rollout buffer of 1024*N_ENVS
#     transitions per update -- enough for stable advantage estimates without
#     needing a huge amount of memory for the (64,64,1) image observations.
#     batch_size=256 and n_epochs=10 are standard PPO settings for
#     continuous-control problems of this size.
#
#   ent_coef = 0.005
#     A small entropy bonus. The reward function includes anti-oscillation /
#     anti-camping penalties (`revisit_pen`) and an action-smoothness penalty
#     (`_smooth_coef`) that a purely exploitative policy could satisfy by
#     barely moving at all early in training, before it has discovered that
#     tracing pays much more. A small entropy bonus keeps exploration alive
#     long enough to find the real (coverage-paying) optimum.
#
#   clip_range = 0.2, vf_coef = 0.5, max_grad_norm = 0.5
#     Standard, well-tested PPO defaults; no environment-specific reason to
#     deviate from them.

TOTAL_TIMESTEPS = 1_000_000

PPO_KWARGS = dict(
    policy="MultiInputPolicy",
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=256,
    n_epochs=10,
    gamma=0.995,
    gae_lambda=0.95,
    clip_range=0.2,
    clip_range_vf=None,
    normalize_advantage=True,
    ent_coef=0.005,
    vf_coef=0.5,
    max_grad_norm=0.5,
    target_kl=None,
    verbose=1,
)

# =============================================================================
# Checkpointing
# =============================================================================

# Total-timestep frequency at which a checkpoint is written. train.py divides
# this by N_ENVS internally because SB3's CheckpointCallback counts callback
# invocations, which occur once per vectorized step (i.e. once per N_ENVS
# real environment steps), not once per raw timestep.
CHECKPOINT_FREQ_TIMESTEPS = 50_000
CHECKPOINT_NAME_PREFIX = "bead_ppo"

# =============================================================================
# Evaluation
# =============================================================================

# Number of deterministic, evenly-spaced contour starting points (from the
# environment's own env.eval_start_points()) used by evaluate.py.
EVAL_N_START_POINTS = 20

# Cap on episode length during evaluation, purely to keep evaluation runtime
# bounded -- the environment's own max_steps/stag_limit already provide a
# natural cutoff, this only prevents a pathological policy from running for
# the full 2500-step budget on every one of EVAL_N_START_POINTS episodes.
EVAL_MAX_STEPS = ENV_KWARGS["max_steps"]

# =============================================================================
# Neural network architecture
# =============================================================================
# The observation is a Dict:
#   "local_map": (64, 64, 1) uint8 -- a local crop of the (covered-curve-
#                erased) distance-transform map, centred on the bead. This is
#                genuinely spatial information (where is the nearby contour,
#                which parts of it are already traced) so it is processed
#                with a small CNN.
#   "state":     (11,) float32 -- position, velocity, direction-to-nearest-
#                uncovered-pixel, normalised distances, coverage fraction and
#                previous action. This is low-dimensional numeric state, so
#                it is processed with a small MLP.
#
# SB3's default MultiInputPolicy would already route the (64,64,1) image
# through its NatureCNN (built for 84x84x4 stacked Atari frames) and the
# state vector through a Flatten no-op before concatenation. NatureCNN is
# oversized for a single-channel 64x64 local crop, so a smaller, purpose-
# built CNN is used instead, fused with the MLP branch by one linear+ReLU
# layer. See README.md "Neural-network architecture" for the full
# justification and a discussion of the size trade-off.
#
# NOTE on image preprocessing: SB3 automatically divides uint8 image
# subspaces of a Dict observation by 255 and (via VecTransposeImage, applied
# automatically to channel-last image subspaces when the vec env is wrapped
# by PPO) transposes (64,64,1) to (1,64,64) *before* this extractor's
# forward() ever sees it. Nothing about that preprocessing lives in env.py.


class BeadFeaturesExtractor(BaseFeaturesExtractor):
    """CNN (local_map) + MLP (state) -> fusion layer -> features_dim vector.

    Kept intentionally small: ~92k CNN params + ~5k MLP params + ~53k fusion
    params, well within "trains on a normal machine" for a few million PPO
    timesteps on CPU or a single consumer GPU.
    """

    def __init__(self, observation_space: spaces.Dict, cnn_out: int = 128,
                 mlp_out: int = 64, features_dim: int = 256):
        super().__init__(observation_space, features_dim=features_dim)

        # By the time the policy builds this extractor, PPO has already
        # wrapped the vec env with VecTransposeImage (because local_map is a
        # channels-last uint8 image space), so this shape is channel-first:
        # (channels, height, width).
        n_input_channels = observation_space["local_map"].shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            sample = th.zeros(1, *observation_space["local_map"].shape)
            n_flatten = self.cnn(sample).shape[1]
        self.cnn_head = nn.Sequential(nn.Linear(n_flatten, cnn_out), nn.ReLU())

        state_dim = observation_space["state"].shape[0]
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, mlp_out), nn.ReLU(),
            nn.Linear(mlp_out, mlp_out), nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(cnn_out + mlp_out, features_dim), nn.ReLU()
        )

    def forward(self, observations):
        image_features = self.cnn_head(self.cnn(observations["local_map"]))
        state_features = self.mlp(observations["state"])
        combined = th.cat([image_features, state_features], dim=1)
        return self.fusion(combined)


# Separate, modestly-sized MLP heads on top of the fused 256-dim feature
# vector for the policy (pi) and value (vf) networks.
POLICY_KWARGS = dict(
    features_extractor_class=BeadFeaturesExtractor,
    features_extractor_kwargs=dict(cnn_out=128, mlp_out=64, features_dim=256),
    net_arch=dict(pi=[128, 128], vf=[128, 128]),
    activation_fn=nn.ReLU,
)
