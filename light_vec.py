"""LightSubprocVecEnv: SB3's SubprocVecEnv, but workers run light_env.light_worker (no torch import in workers).
Import this only from the main process."""
import multiprocessing as mp

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

import light_env


class LightSubprocVecEnv(SubprocVecEnv):
    def __init__(self, env_fns):
        self.waiting = False
        self.closed = False
        ctx = mp.get_context("spawn")
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in env_fns])
        self.processes = []
        for work_remote, remote, fn in zip(self.work_remotes, self.remotes, env_fns):
            p = ctx.Process(target=light_env.light_worker, args=(work_remote, remote, fn), daemon=True)
            p.start()
            self.processes.append(p)
            work_remote.close()
        self.remotes[0].send(("get_spaces", None))
        obs_space, act_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), obs_space, act_space)
