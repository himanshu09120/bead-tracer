"""
run_experiment.py -- resumable PPO training for one (obs_mode, seed, phi).

Writes ONLY to runs/<name>/ (never models/ or checkpoints/):
    runs/<name>/checkpoints/bead_ppo_<k>_steps.zip   every 50k steps
    runs/<name>/final.zip                            after the last full rollout (1,003,520 steps)
    runs/<name>/monitor/                             SB3 Monitor csv
Hyperparameters and env kwargs come from config.py unchanged.
Re-running skips a finished run and resumes an interrupted one from its latest checkpoint.

    python run_experiment.py --obs-mode no_direction --seed 1
"""
import argparse, glob, json, os, re, time

import torch
from stable_baselines3 import PPO

import config, train

ROOT = os.path.join(config.PROJECT_ROOT, "runs")


def run_name(obs_mode, seed, phi):
    return f"{obs_mode}_s{seed}" + ("" if phi == 0.95 else f"_phi{phi}")


def latest_checkpoint(ckpt_dir):
    best, best_k = None, -1
    for f in glob.glob(os.path.join(ckpt_dir, "bead_ppo_*_steps.zip")):
        k = int(re.search(r"_(\d+)_steps", f).group(1))
        if k > best_k:
            best, best_k = f, k
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--obs-mode", default="full")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--phi", type=float, default=0.95)
    p.add_argument("--image", default=config.DEFAULT_IMAGE_PATH)
    p.add_argument("--timesteps", type=int, default=config.TOTAL_TIMESTEPS)
    p.add_argument("--n-envs", type=int, default=config.N_ENVS)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--name", default=None)
    a = p.parse_args()

    name = a.name or run_name(a.obs_mode, a.seed, a.phi)
    d = os.path.join(ROOT, name)
    ck = os.path.join(d, "checkpoints")
    final = os.path.join(d, "final.zip")
    if os.path.exists(final):
        print(f"[skip] {name}: final.zip exists"); return
    os.makedirs(ck, exist_ok=True)
    torch.set_num_threads(a.threads)

    env_kwargs = dict(config.ENV_KWARGS, completion_frac=a.phi, obs_mode=a.obs_mode)
    json.dump(dict(vars(a), env_kwargs=env_kwargs, ppo_kwargs={k: str(v) for k, v in config.PPO_KWARGS.items()}),
              open(os.path.join(d, "run.json"), "w"), indent=1)

    vec = train.build_vec_env(a.image, a.n_envs, a.seed, os.path.join(d, "monitor"), env_kwargs, True)
    resume = latest_checkpoint(ck)
    if resume:
        print(f"[resume] {name} from {resume}")
        model = PPO.load(resume, env=vec, device=a.device)
    else:
        kw = dict(config.PPO_KWARGS, device=a.device)
        model = train.build_model(vec, seed=a.seed, tensorboard_log=os.path.join(d, "tb"), ppo_kwargs=kw)
    cbs = train.build_callbacks(a.n_envs, ck, config.CHECKPOINT_FREQ_TIMESTEPS, config.CHECKPOINT_NAME_PREFIX)
    t = time.time()
    model.learn(total_timesteps=max(a.timesteps - model.num_timesteps, 1), callback=cbs,
                tb_log_name="ppo", reset_num_timesteps=resume is None)
    model.save(final)
    json.dump(dict(num_timesteps=model.num_timesteps, wall_s=time.time() - t),
              open(os.path.join(d, "done.json"), "w"))
    vec.close()
    print(f"[done] {name} {model.num_timesteps} steps")


if __name__ == "__main__":
    main()
