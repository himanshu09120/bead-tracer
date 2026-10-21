#!/usr/bin/env bash
# Locked evaluation worker: repeatedly claims any finished-but-unevaluated run (mkdir lock) and evaluates it.
# Exits when every run has .eval_done.
export PYTHONPATH=.
RUNS="no_direction:42 no_direction:1 no_direction:2 no_direction:3 no_direction:4 full:1 full:2 full:3 full:4 no_direction_noisy:42 no_direction_noisy:1 no_direction_noisy:2 no_visual:42 no_visual:1 no_visual:2"
while true; do
  left=0; claimed=0
  for spec in $RUNS; do
    t=${spec%%:*}; s=${spec##*:}; d=runs/${t}_s$s
    [ -f $d/.eval_done ] && continue
    left=1
    [ -f $d/final.zip ] || continue
    mkdir $d/.lock 2>/dev/null || continue
    claimed=1
    if python evaluate_run.py --run ${t}_s$s --tier $t >> runs/logs/eval_${t}_s$s.log 2>&1; then touch $d/.eval_done; else rmdir $d/.lock; sleep 30; fi
  done
  [ $left = 0 ] && break
  [ $claimed = 0 ] && sleep 60
done
