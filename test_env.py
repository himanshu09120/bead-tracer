"""
test_env.py -- Two independent test suites, run together or separately.

1. ENVIRONMENT SELF-TESTS (--env-only): the environment's own, pre-existing
   coverage-metric test suite (env._run_tests(), the same code that runs via
   `python env.py`). Preserved and exposed here unmodified so it's easy to
   run as part of this project's testing story.

2. RL SMOKE TEST (--rl-only): verifies the SB3/PPO side actually works end
   to end against BeadEnv -- environment creation, PPO initialization,
   observation/action compatibility, a short training run, model saving,
   model loading, inference, and evaluation. Runs against the environment's
   own tiny synthetic circle target (env._make_env()) by default so it
   finishes in well under a minute; pass --image to smoke-test against a
   real target image instead.

Usage:
    python test_env.py                # both suites
    python test_env.py --env-only
    python test_env.py --rl-only
    python test_env.py --rl-only --image target_images/path1.png --smoke-timesteps 2000
"""

import argparse
import os
import shutil
import tempfile

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import config
import env as env_module
from evaluate import run_episode
from train import build_model, load_target_image


# =============================================================================
# 1. Environment self-tests (preserved, unmodified)
# =============================================================================

def run_env_self_tests() -> bool:
    print("#" * 78)
    print("# SUITE 1/2: BeadEnv's own built-in coverage-metric self-tests")
    print("#" * 78)
    ok = env_module._run_tests()
    # The self-test suite writes a few PNGs into the working directory as a
    # side effect (coverage_test_*.png) -- tidy those up so running the test
    # suite doesn't litter the project root.
    for fname in ("coverage_test_revisit.png", "coverage_test_partial.png",
                  "coverage_test_full.png"):
        if os.path.isfile(fname):
            os.remove(fname)
    return ok


# =============================================================================
# 2. RL smoke test
# =============================================================================

def _smoke_env(image_path: str = None):
    """A single BeadEnv instance for the smoke test: either the environment's
    own tiny synthetic circle target (fast, no dependency on a real target
    image) or a real image if --image was passed."""
    if image_path is None:
        return env_module._make_env(max_steps=300, stag_limit=300)
    img = load_target_image(image_path)
    return env_module.BeadEnv(target_image=img, max_steps=300, stag_limit=300)


def run_rl_smoke_test(image_path: str = None, smoke_timesteps: int = 1000,
                       seed: int = 0) -> bool:
    print("#" * 78)
    print("# SUITE 2/2: RL smoke test (SB3 PPO x BeadEnv)")
    print("#" * 78)

    checks = []

    def check(name, ok, detail=""):
        checks.append((name, ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))

    tmp_dir = tempfile.mkdtemp(prefix="bead_rl_smoke_")
    try:
        # -- 1. environment creation -----------------------------------------
        raw_env = _smoke_env(image_path)
        check("environment creation", isinstance(raw_env, env_module.BeadEnv),
              f"{raw_env.total_contour_pixels} contour pixels")

        # -- 2. Gym API validation via SB3's own checker ----------------------
        try:
            check_env(raw_env, warn=True, skip_render_check=True)
            check("stable_baselines3 check_env validation", True)
        except Exception as e:  # noqa: BLE001
            check("stable_baselines3 check_env validation", False, str(e))
            raise

        # -- vec env for PPO ---------------------------------------------------
        def _make():
            e = _smoke_env(image_path)
            e.action_space.seed(seed)
            return Monitor(e)

        vec_env = DummyVecEnv([_make])

        # -- 3. PPO initialization ---------------------------------------------
        smoke_ppo_kwargs = dict(config.PPO_KWARGS)
        smoke_ppo_kwargs.update(
            n_steps=64, batch_size=32, n_epochs=2, device="auto", verbose=0,
        )
        model = build_model(vec_env, seed=seed, tensorboard_log=None,
                             ppo_kwargs=smoke_ppo_kwargs)
        check("PPO initialization (MultiInputPolicy + BeadFeaturesExtractor)", True,
              f"{sum(p.numel() for p in model.policy.parameters()):,} params")

        # -- 4. observation compatibility --------------------------------------
        obs = vec_env.reset()
        try:
            action, _ = model.predict(obs, deterministic=True)
            check("observation compatibility (model.predict on reset obs)", True)
        except Exception as e:  # noqa: BLE001
            check("observation compatibility (model.predict on reset obs)", False, str(e))
            raise

        # -- 5. action compatibility --------------------------------------------
        try:
            assert vec_env.action_space.contains(action[0])
            obs, reward, done, info = vec_env.step(action)
            check("action compatibility (env.step on predicted action)", True,
                  f"reward={float(reward[0]):+.3f}")
        except Exception as e:  # noqa: BLE001
            check("action compatibility (env.step on predicted action)", False, str(e))
            raise

        # -- 6. short training ----------------------------------------------------
        try:
            model.learn(total_timesteps=smoke_timesteps, progress_bar=False)
            check("short training run", True, f"{smoke_timesteps} timesteps")
        except Exception as e:  # noqa: BLE001
            check("short training run", False, str(e))
            raise

        # -- 7. model saving -------------------------------------------------------
        save_path = os.path.join(tmp_dir, "smoke_model.zip")
        model.save(save_path)
        check("model saving", os.path.isfile(save_path), save_path)

        # -- 8. model loading -------------------------------------------------------
        loaded_model = PPO.load(save_path, device="auto")
        check("model loading", loaded_model is not None)

        # -- 9. inference ------------------------------------------------------------
        eval_env = _smoke_env(image_path)
        obs, info = eval_env.reset(seed=seed)
        action, _ = loaded_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        check("inference (deterministic predict + step on loaded model)", True,
              f"coverage_pct={info['coverage_pct']:.3f}%")

        # -- 10. evaluation (env.eval_start_points + evaluate.run_episode) --------
        start_points = eval_env.eval_start_points(2)
        result = run_episode(eval_env, loaded_model, start_points[0],
                              deterministic=True, max_steps=100)
        required_keys = {"coverage", "coverage_pct", "covered_pixels",
                          "total_contour_pixels", "reward", "steps"}
        check("evaluation (evaluate.run_episode on a fixed_start point)",
              required_keys.issubset(result.keys()),
              f"coverage_pct={result['coverage_pct']:.3f}% steps={result['steps']}")

        vec_env.close()
        eval_env.close()
        raw_env.close()

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    n_pass = sum(1 for _, ok in checks if ok)
    print(f"\n  {n_pass}/{len(checks)} RL smoke-test checks passed")
    return n_pass == len(checks)


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Run BeadEnv self-tests and/or the RL smoke test.")
    p.add_argument("--env-only", action="store_true", help="Run only the environment self-tests.")
    p.add_argument("--rl-only", action="store_true", help="Run only the RL smoke test.")
    p.add_argument("--image", type=str, default=None,
                    help="Real target image to use for the RL smoke test "
                         "(default: the environment's own synthetic circle target).")
    p.add_argument("--smoke-timesteps", type=int, default=1000,
                    help="Timesteps for the RL smoke test's short training run.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    run_env = not args.rl_only
    run_rl = not args.env_only

    results = {}
    if run_env:
        results["env_self_tests"] = run_env_self_tests()
    if run_rl:
        results["rl_smoke_test"] = run_rl_smoke_test(
            image_path=args.image, smoke_timesteps=args.smoke_timesteps, seed=args.seed,
        )

    print("\n" + "#" * 78)
    print("# TEST SUMMARY")
    print("#" * 78)
    for name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    all_ok = all(results.values())
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
