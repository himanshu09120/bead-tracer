"""
light_env.py -- env factory for SubprocVecEnv workers that does NOT import torch / stable-baselines3.

On Windows (spawn) every worker re-imports the modules its factory lives in; with train.py/config.py that meant
~470 MB of torch per worker.  This module only needs cv2, numpy, gymnasium and env.py (~100 MB).
EpisodeStats reproduces what SB3's Monitor puts in info["episode"] ({"r", "l", "t"}), which is all SB3 and
train.CoverageLoggingCallback read; the environment itself is unchanged.
"""
import time

import cv2
import gymnasium as gym

from env import BeadEnv


class EpisodeStats(gym.Wrapper):
    def reset(self, **kw):
        self._r, self._l, self._t0 = 0.0, 0, time.time()
        return self.env.reset(**kw)

    def step(self, a):
        obs, r, te, tr, info = self.env.step(a)
        self._r += float(r)
        self._l += 1
        if te or tr:
            info = dict(info, episode={"r": self._r, "l": self._l, "t": round(time.time() - self._t0, 6)})
        return obs, r, te, tr, info


class EnvFn:
    """Picklable factory (pickled by reference, so workers import only this module)."""

    def __init__(self, image_path, seed, rank, env_kwargs, curriculum=True):
        self.args = (image_path, seed, rank, env_kwargs, curriculum)

    def __call__(self):
        image_path, seed, rank, env_kwargs, curriculum = self.args
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(image_path)
        env = BeadEnv(target_image=img, curriculum=curriculum, **env_kwargs)
        env.action_space.seed(seed + rank)
        env = EpisodeStats(env)
        env.reset(seed=seed + rank)
        return env


def _is_wrapped(env, cls):
    while isinstance(env, gym.Wrapper):
        if isinstance(env, cls):
            return True
        env = env.env
    return False


def light_worker(remote, parent_remote, env_fn):
    """Port of stable_baselines3.common.vec_env.subproc_vec_env._worker (SB3 2.9.0) that lives in this torch-free
    module, so spawned workers never import SB3/torch.  Same protocol, same auto-reset semantics."""
    parent_remote.close()
    env = env_fn()
    reset_info = {}
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, reward, terminated, truncated, info = env.step(data)
                done = terminated or truncated
                info["TimeLimit.truncated"] = truncated and not terminated
                if done:
                    info["terminal_observation"] = obs
                    obs, reset_info = env.reset()
                remote.send((obs, reward, done, info, reset_info))
            elif cmd == "reset":
                opts = {"options": data[1]} if data[1] else {}
                obs, reset_info = env.reset(seed=data[0], **opts)
                remote.send((obs, reset_info))
            elif cmd == "render":
                remote.send(env.render())
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                remote.send(env.get_wrapper_attr(data[0])(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(env.get_wrapper_attr(data))
            elif cmd == "has_attr":
                try:
                    env.get_wrapper_attr(data)
                    remote.send(True)
                except AttributeError:
                    remote.send(False)
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))
            elif cmd == "is_wrapped":
                remote.send(_is_wrapped(env, data))
            else:
                raise NotImplementedError(cmd)
        except (EOFError, KeyboardInterrupt):
            break
