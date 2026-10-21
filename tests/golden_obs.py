"""Hash of a fixed rollout of BeadEnv in obs_mode='full' (golden value captured from the pre-obs_mode env)."""
import hashlib, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np
import config
from env import BeadEnv

def rollout_hash(**kw):
    env = BeadEnv(cv2.imread("target_images/path1.png"), curriculum=True, **config.ENV_KWARGS, **kw)
    rng = np.random.default_rng(0)
    h = hashlib.sha256()
    obs, _ = env.reset(seed=3)
    for t in range(600):
        for k in ("local_map", "state"):
            h.update(obs[k].tobytes())
        obs, r, te, tr, _ = env.step(rng.uniform(-1, 1, 2).astype(np.float32))
        h.update(np.float64(r).tobytes())
        if te or tr:
            obs, _ = env.reset(seed=t)
    return h.hexdigest()

if __name__ == "__main__":
    print(rollout_hash())
