"""
Reproducibility regression tests (slow: ~6 evaluations of 20 episodes; run `pytest tests -m slow` or all).

1. FROZEN: harness numbers (CPU, 1 torch thread, test starts, phi=0.95) in tests/reference_values.json must be
   reproduced to 0.05 -- guards refactors.
2. VERIFIED (author's numbers from the original machine): path2/path3 are reproduced to 0.05; path1 only to
   0.3 pp coverage / 1 completion because path1 rollouts are chaotic and the original run used other
   thread/device settings.  This gap is a documented finding, not something to tighten silently.
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from evaluation import evaluate, PPOPolicy

MODELS = {"final": "models/bead_ppo_final.zip", "m163258": "models/bead_ppo_20260819_163258_final.zip"}
# (coverage_mean, n_complete) quoted by the author
VERIFIED = {
    "final|path1": (86.18, 0), "m163258|path1": (94.84, 12),
    "final|path2": (56.6, 0), "final|path3": (50.5, 0),
    "m163258|path2": (57.8, 0), "m163258|path3": (63.8, 0),
}
_cache = {}


def result(key):
    if key not in _cache:
        m, t = key.split("|")
        _cache[key] = evaluate(PPOPolicy(MODELS[m]), t, "test", keep_traj=False)["summary"]
    return _cache[key]


REF = json.load(open(os.path.join(ROOT, "tests", "reference_values.json")))


@pytest.mark.slow
@pytest.mark.parametrize("key", sorted(REF))
def test_frozen_reference(key):
    s, r = result(key), REF[key]
    assert s["coverage_mean"] == pytest.approx(r["coverage_mean"], abs=0.05)
    assert s["n_complete"] == r["n_complete"]
    assert s["ret_mean"] == pytest.approx(r["ret_mean"], abs=0.05)
    assert s["steps_mean"] == pytest.approx(r["steps_mean"], abs=0.05)


@pytest.mark.slow
@pytest.mark.parametrize("key", sorted(VERIFIED))
def test_author_verified(key):
    cov, comp = VERIFIED[key]
    s = result(key)
    tol = 0.05 if not key.endswith("path1") else 0.3
    assert s["coverage_mean"] == pytest.approx(cov, abs=tol)
    assert abs(s["n_complete"] - comp) <= (0 if not key.endswith("path1") else 1)
