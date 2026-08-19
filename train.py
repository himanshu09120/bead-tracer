"""
train.py -- Trains a Stable-Baselines3 PPO agent on the (unmodified) BeadEnv.

Everything RL/model-side lives here and in config.py; env.py is imported and
used strictly through its existing public interface (constructor kwargs,
reset(), step(), eval_start_points(), render_coverage()) -- nothing about the
environment is changed to make this work.

Usage:
    python train.py
    python train.py --image target_images/path1.png --timesteps 2000000
    python train.py --n-envs 8 --seed 123 --run-name my_run
    python train.py --resume checkpoints/bead_ppo_500000_steps.zip
"""

import argparse
import os
import time

import cv2
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import config
from env import BeadEnv


# =============================================================================
# Environment construction
# =============================================================================

def load_target_image(image_path: str) -> np.ndarray:
    """Loads a target image from disk for BeadEnv. Raises a clear error
    (rather than letting cv2 silently hand BeadEnv a None array) when the
    path doesn't exist or isn't a readable image, and reminds the caller
    that every script in this project accepts --image on the command line."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Target image not found: {image_path}\n"
            f"Pass a real image with --image <path>, e.g.:\n"
            f"  python train.py --image target_images/path1.png"
        )
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image (unsupported/corrupt file): {image_path}")
    return img


def make_bead_env_fn(image_path: str, seed: int, rank: int, monitor_dir: str,
                      env_kwargs: dict, curriculum: bool = True):
    """Returns a thunk that builds one Monitor-wrapped BeadEnv instance. Used
    as the per-worker factory for DummyVecEnv/SubprocVecEnv -- the image is
    read from disk independently inside each thunk so it works unchanged
    whether the vec env runs in-process or in subprocesses."""

    def _init():
        img = load_target_image(image_path)
        env = BeadEnv(target_image=img, curriculum=curriculum, **env_kwargs)
        env.action_space.seed(seed + rank)
        os.makedirs(monitor_dir, exist_ok=True)
        env = Monitor(env, filename=os.path.join(monitor_dir, f"env_{rank}"))
        obs, _ = env.reset(seed=seed + rank)
        return env

    return _init


def build_vec_env(image_path: str, n_envs: int, seed: int, monitor_dir: str,
                   env_kwargs: dict, use_subproc: bool = True, curriculum: bool = True):
    """Builds an n_envs-way vectorized, Monitor-wrapped BeadEnv. Training
    keeps the environment's own randomized curriculum spawn (curriculum=True)
    -- only evaluate.py/inference.py force deterministic fixed_start resets,
    per the project's separation of "randomized training" vs "deterministic
    evaluation"."""
    env_fns = [
        make_bead_env_fn(image_path, seed, rank, monitor_dir, env_kwargs, curriculum)
        for rank in range(n_envs)
    ]
    vec_cls = SubprocVecEnv if (use_subproc and n_envs > 1) else DummyVecEnv
    vec_env = vec_cls(env_fns)
    return vec_env


def validate_env(image_path: str, env_kwargs: dict) -> bool:
    """Runs Stable-Baselines3's own environment checker against a single
    BeadEnv instance before any training happens. This exercises the Gym API
    contract (obs/action spaces, reset/step signatures, dtypes) without
    touching env.py."""
    from stable_baselines3.common.env_checker import check_env

    print("Validating environment with stable_baselines3.common.env_checker.check_env ...")
    img = load_target_image(image_path)
    env = BeadEnv(target_image=img, **env_kwargs)
    check_env(env, warn=True, skip_render_check=True)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs), "reset() observation not in observation_space"
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs), "step() observation not in observation_space"
    assert isinstance(reward, (int, float, np.floating)), "reward must be a scalar"
    for key in ("coverage", "coverage_pct", "covered_pixels", "total_contour_pixels"):
        assert key in info, f"info missing required coverage key: {key}"
    env.close()
    print(f"Environment OK -- {env.total_contour_pixels} contour pixels in target image.")
    return True


# =============================================================================
# Model construction
# =============================================================================

def build_model(vec_env, seed: int, tensorboard_log: str, ppo_kwargs: dict = None,
                 policy_kwargs: dict = None):
    """Builds a Stable-Baselines3 PPO model with GAE-based advantage
    estimation (SB3's PPO always uses GAE -- gae_lambda is a first-class PPO
    hyperparameter, so no custom advantage code is written here) and the
    Dict-observation CNN+MLP policy from config.py."""
    ppo_kwargs = dict(config.PPO_KWARGS if ppo_kwargs is None else ppo_kwargs)
    policy = ppo_kwargs.pop("policy", "MultiInputPolicy")
    model = PPO(
        policy,
        vec_env,
        seed=seed,
        tensorboard_log=tensorboard_log,
        policy_kwargs=config.POLICY_KWARGS if policy_kwargs is None else policy_kwargs,
        **ppo_kwargs,
    )
    return model


# =============================================================================
# Callbacks
# =============================================================================

class CoverageLoggingCallback(BaseCallback):
    """Surfaces the environment's own coverage_info() through Stable-
    Baselines3's logger (and therefore TensorBoard), without modifying
    env.py. BeadEnv already returns coverage / coverage_pct / covered_pixels
    / total_contour_pixels / task_complete / steps in `info` on every reset()
    and step() call; this callback just reads that dict off the VecEnv step
    results and records the values at episode boundaries so TensorBoard
    shows genuine per-episode final coverage, not a mid-episode snapshot.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if not done:
                continue
            # Monitor's own per-episode summary (present because every env
            # is Monitor-wrapped) -- included for a raw-reward cross-check
            # alongside the coverage numbers.
            ep = info.get("episode")
            if ep is not None:
                self.logger.record_mean("rollout/ep_return_raw", ep["r"])

            if "coverage_pct" in info:
                self.logger.record_mean("rollout/ep_coverage_pct", info["coverage_pct"])
                self.logger.record_mean("rollout/ep_covered_pixels", info["covered_pixels"])
                self.logger.record_mean("rollout/ep_total_contour_pixels", info["total_contour_pixels"])
                self.logger.record_mean("rollout/ep_task_complete_rate", float(info["task_complete"]))
                self.logger.record_mean("rollout/ep_steps", info["steps"])
        return True


def build_callbacks(n_envs: int, checkpoint_dir: str, checkpoint_freq_timesteps: int,
                     name_prefix: str):
    """CheckpointCallback is provided by Stable-Baselines3 -- checkpointing is
    not reimplemented here. save_freq is expressed in "calls to the
    callback", which happen once per vectorized step (i.e. once per n_envs
    real env steps), so it is divided by n_envs to keep the *timestep*
    frequency requested in config.py accurate regardless of n_envs."""
    save_freq = max(checkpoint_freq_timesteps // max(n_envs, 1), 1)
    checkpoint_cb = CheckpointCallback(
        save_freq=save_freq,
        save_path=checkpoint_dir,
        name_prefix=name_prefix,
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=1,
    )
    coverage_cb = CoverageLoggingCallback()
    return CallbackList([checkpoint_cb, coverage_cb])


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Train a PPO agent on BeadEnv.")
    p.add_argument("--image", type=str, default=config.DEFAULT_IMAGE_PATH,
                    help="Path to the target image (default: target_images/path1.png).")
    p.add_argument("--timesteps", type=int, default=config.TOTAL_TIMESTEPS,
                    help="Total training timesteps.")
    p.add_argument("--n-envs", type=int, default=config.N_ENVS,
                    help="Number of parallel environments.")
    p.add_argument("--seed", type=int, default=config.SEED,
                    help="Random seed (NumPy, PyTorch, Gymnasium, SB3, env).")
    p.add_argument("--run-name", type=str, default=None,
                    help="Name for this run's TensorBoard/log subdirectory "
                         "(default: bead_ppo_<timestamp>).")
    p.add_argument("--resume", type=str, default=None,
                    help="Path to an existing model/checkpoint .zip to resume training from.")
    p.add_argument("--no-subproc", action="store_true",
                    help="Use DummyVecEnv (single process) instead of SubprocVecEnv.")
    p.add_argument("--device", type=str, default="auto",
                    help="'auto', 'cpu', or 'cuda'.")
    p.add_argument("--skip-validation", action="store_true",
                    help="Skip the check_env() validation pass before training.")
    return p.parse_args()


def main():
    args = parse_args()

    run_name = args.run_name or f"bead_ppo_{time.strftime('%Y%m%d_%H%M%S')}"
    print(f"Run name: {run_name}")

    # -- Reproducibility: seeds NumPy, PyTorch and Python's random module. --
    set_random_seed(args.seed)

    # -- Validate the environment before spending any compute on training. --
    if not args.skip_validation:
        validate_env(args.image, config.ENV_KWARGS)

    monitor_dir = os.path.join(config.LOGS_DIR, run_name)
    os.makedirs(monitor_dir, exist_ok=True)

    vec_env = build_vec_env(
        image_path=args.image,
        n_envs=args.n_envs,
        seed=args.seed,
        monitor_dir=monitor_dir,
        env_kwargs=config.ENV_KWARGS,
        use_subproc=(not args.no_subproc) and config.USE_SUBPROC,
        curriculum=True,   # training keeps the environment's randomized spawn behavior
    )
    # Each sub-env is already individually Monitor-wrapped (in
    # make_bead_env_fn), which is what puts an "episode" entry into `info` at
    # episode boundaries -- SB3 reads that directly off the VecEnv's per-env
    # infos to compute rollout/ep_rew_mean and rollout/ep_len_mean, and
    # CoverageLoggingCallback reads the same infos for the coverage metrics.
    # A further VecMonitor wrapper is unnecessary here and would only
    # duplicate/overwrite those per-env Monitor statistics.

    tb_log_dir = config.TENSORBOARD_DIR

    if args.resume:
        print(f"Resuming training from {args.resume}")
        model = PPO.load(args.resume, env=vec_env, device=args.device,
                          tensorboard_log=tb_log_dir)
    else:
        ppo_kwargs = dict(config.PPO_KWARGS)
        ppo_kwargs["device"] = args.device
        model = build_model(vec_env, seed=args.seed, tensorboard_log=tb_log_dir,
                             ppo_kwargs=ppo_kwargs)

    print(model.policy)

    callbacks = build_callbacks(
        n_envs=args.n_envs,
        checkpoint_dir=config.CHECKPOINTS_DIR,
        checkpoint_freq_timesteps=config.CHECKPOINT_FREQ_TIMESTEPS,
        name_prefix=config.CHECKPOINT_NAME_PREFIX,
    )

    print(f"Training for {args.timesteps:,} timesteps across {args.n_envs} env(s) ...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        tb_log_name=run_name,
        reset_num_timesteps=(args.resume is None),
        progress_bar=True,
    )

    model.save(config.FINAL_MODEL_PATH)
    print(f"Saved final model to {config.FINAL_MODEL_PATH}")

    timestamped_path = os.path.join(config.MODELS_DIR, f"{run_name}_final.zip")
    model.save(timestamped_path)
    print(f"Saved timestamped copy to {timestamped_path}")

    vec_env.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
