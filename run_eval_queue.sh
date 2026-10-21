#!/usr/bin/env bash
# Evaluate every run (in training-queue order) as soon as its final.zip exists. Resumable (evaluate_run skips done files).
export PYTHONPATH=.
mkdir -p runs/logs
for spec in "no_direction 42" "no_direction 1" "no_direction 2" "no_direction 3" "no_direction 4" \
            "full 1" "full 2" "full 3" "full 4" \
            "no_direction_noisy 42" "no_direction_noisy 1" "no_direction_noisy 2" \
            "no_visual 42" "no_visual 1" "no_visual 2"; do
  set -- $spec
  until [ -f runs/$1_s$2/final.zip ]; do sleep 60; done
  python evaluate_run.py --run $1_s$2 --tier $1 >> runs/logs/eval_$1_s$2.log 2>&1
done
