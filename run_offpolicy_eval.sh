#!/usr/bin/env bash
# Evaluate the six FINAL SAC/TD3 models (no checkpoint selection) on the 20 test starts of all 13 targets
# (CPU, 1 torch thread, deterministic, phi=0.95).  Existing raw results are skipped, so it is resumable.
# Raw: results/raw/<algo>_final__full__s<seed>__<target>__test.json
mkdir -p runs/logs
for s in 42 1 2; do for a in sac td3; do
  PYTHONPATH=. python evaluate_model.py --algo $a --method ${a}_final --label ${a}_s$s --seed $s --train-phi 0.95 \
    --model runs/${a}_s$s/final.zip > runs/logs/eval_${a}_s$s.log 2>&1
done; done
python -m analysis.sac_td3
