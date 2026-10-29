"""
run_offpolicy.py -- SAC / TD3 training for one (algo, seed), the off-policy counterpart of run_experiment.py.

Same environment (light_env.EnvFn, path1, obs_mode="full", config.ENV_KWARGS, curriculum spawn, 4 workers), same
CNN+MLP features extractor and pi/q head widths (config.POLICY_KWARGS), same 1,000,000 timesteps.  Only the
algorithm changes; the algorithm-specific hyperparameters are in OFFPOLICY_KWARGS below and are documented in
SAC_TD3_PAPER_CHANGES.md.  No checkpoint selection: final.zip is the result.

Writes ONLY to runs/<algo>_s<seed>/ (final.zip, run.json, done.json, tb/, ckpt/ every 100k for diagnosis only).
Skips a finished run; an interrupted run restarts from scratch (replay buffer is not persisted).

    python run_offpolicy.py --algo sac --seed 42
"""
import argparse, json, os, time

from light_env import EnvFn

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(PROJECT_ROOT, "runs")

# Shared by SAC and TD3.  gamma matches PPO; lr matches PPO's 3e-4.  buffer_size is capped by RAM: every transition
# stores two 64x64 uint8 crops (SB3's DictReplayBuffer keeps obs and next_obs), ~8.3 kB -> 200k = ~1.7 GB per run.
COMMON = dict(learning_rate=3e-4, buffer_size=200_000, learning_starts=10_000, batch_size=256, tau=0.005,
              gamma=0.995, train_freq=1)
OFFPOLICY_KWARGS = {
    "sac": dict(COMMON, ent_coef="auto"),
    "td3": dict(COMMON, learning_rate=1e-4, policy_delay=2, target_policy_noise=0.2, target_noise_clip=0.5),  # + action noise below
}
TD3_ACTION_NOISE_SIGMA = 0.3   # Gaussian exploration noise.  lr 1e-4 / sigma 0.3 chosen on TRAINING coverage after the
                               # defaults (lr 3e-4, sigma 0.1) collapsed to a saturated corner action (see changes doc)


def main():
    import numpy as np
    import torch
    from stable_baselines3 import SAC, TD3
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback
    from light_vec import LightSubprocVecEnv
    import config, train

    p = argparse.ArgumentParser()
    p.add_argument("--algo", required=True, choices=["sac", "td3"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--image", default=config.DEFAULT_IMAGE_PATH)
    p.add_argument("--timesteps", type=int, default=config.TOTAL_TIMESTEPS)
    p.add_argument("--n-envs", type=int, default=config.N_ENVS)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--gradient-steps", type=int, default=None, help="per vec step; default 1 = SB3 default (1 update per vec step = 1 per n_envs transitions)")
    p.add_argument("--name", default=None)
    p.add_argument("--lr", type=float, default=None, help="override COMMON learning_rate")
    p.add_argument("--noise-sigma", type=float, default=None, help="TD3 action-noise sigma override")
    p.add_argument("--learning-starts", type=int, default=None)
    a = p.parse_args()

    name = a.name or f"{a.algo}_s{a.seed}"
    d = os.path.join(ROOT, name)
    final = os.path.join(d, "final.zip")
    if os.path.exists(final):
        print(f"[skip] {name}: final.zip exists"); return
    os.makedirs(os.path.join(d, "ckpt"), exist_ok=True)
    torch.set_num_threads(a.threads)

    env_kwargs = dict(config.ENV_KWARGS, completion_frac=0.95, obs_mode="full")
    kw = dict(OFFPOLICY_KWARGS[a.algo], gradient_steps=a.gradient_steps or 1)
    if a.lr: kw["learning_rate"] = a.lr
    if a.learning_starts: kw["learning_starts"] = a.learning_starts
    sigma = a.noise_sigma or TD3_ACTION_NOISE_SIGMA
    json.dump(dict(vars(a), env_kwargs=env_kwargs, algo_kwargs=kw, td3_action_noise_sigma=sigma,
                   policy_kwargs={k: str(v) for k, v in config.POLICY_KWARGS.items()},
                   versions=dict(torch=torch.__version__)), open(os.path.join(d, "run.json"), "w"), indent=1)

    fns = [EnvFn(a.image, a.seed, rank, env_kwargs) for rank in range(a.n_envs)]
    vec = (LightSubprocVecEnv if a.n_envs > 1 else DummyVecEnv)(fns)
    pk = dict(config.POLICY_KWARGS, net_arch=dict(pi=[128, 128], qf=[128, 128]))   # q-nets mirror PPO's vf widths
    algo_cls = {"sac": SAC, "td3": TD3}[a.algo]
    extra = {}
    if a.algo == "td3":
        extra["action_noise"] = NormalActionNoise(np.zeros(2), sigma * np.ones(2))
    model = algo_cls("MultiInputPolicy", vec, seed=a.seed, device=a.device, policy_kwargs=pk, verbose=1,
                     tensorboard_log=os.path.join(d, "tb"), **kw, **extra)
    cbs = [CheckpointCallback(max(100_000 // a.n_envs, 1), os.path.join(d, "ckpt"), name_prefix=a.algo),
           train.CoverageLoggingCallback()]
    t = time.time()
    model.learn(total_timesteps=a.timesteps, callback=cbs, tb_log_name=a.algo)
    model.save(final)
    json.dump(dict(num_timesteps=model.num_timesteps, wall_s=time.time() - t), open(os.path.join(d, "done.json"), "w"))
    vec.close()
    print(f"[done] {name} {model.num_timesteps} steps")


if __name__ == "__main__":
    main()
