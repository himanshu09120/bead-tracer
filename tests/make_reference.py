"""Freeze regression reference values (harness: CPU, torch 1 thread, test starts, phi=0.95, deterministic)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation import evaluate, PPOPolicy

MODELS = {"final": "models/bead_ppo_final.zip", "m163258": "models/bead_ppo_20260819_163258_final.zip"}
out = {}
for k, m in MODELS.items():
    pol = PPOPolicy(m)
    for t in ("path1", "path2", "path3"):
        s = evaluate(pol, t, "test", keep_traj=False)["summary"]
        out[f"{k}|{t}"] = {x: s[x] for x in ("coverage_mean", "coverage_sd", "n_complete", "ret_mean", "steps_mean")}
        print(k, t, out[f"{k}|{t}"], flush=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "reference_values.json"), "w"), indent=1)
