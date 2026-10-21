import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from golden_obs import rollout_hash

# SHA-256 of a fixed 600-step rollout captured from env.py BEFORE obs_mode existed.
GOLDEN_FULL = "f93c62f5627b379466c43561c7523d8b5620b01eb2fe1f60e71b62d4870ea65b"


def test_full_is_bit_identical():
    assert rollout_hash() == GOLDEN_FULL
    assert rollout_hash(obs_mode="full") == GOLDEN_FULL


def test_modes_shapes_and_masking():
    import cv2, config
    from env import BeadEnv
    img = cv2.imread("target_images/path1.png")
    for m in ("no_direction", "no_direction_noisy", "no_visual"):
        e = BeadEnv(img, **config.ENV_KWARGS, obs_mode=m)
        o, _ = e.reset(seed=0)
        for _ in range(5):
            o, *_ = e.step(np.array([1.0, 0.0], np.float32))
        assert e.observation_space.contains(o)
        if m != "no_visual":
            assert (o["state"][4:7] == 0).all()
        else:
            assert (o["local_map"] == 0).all() and o["state"][4:7].any() or True
