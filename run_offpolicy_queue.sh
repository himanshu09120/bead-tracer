#!/usr/bin/env bash
# Train SAC and TD3 (seeds 42, 1, 2) on path1, P at a time (default 2: each run holds a ~2 GB replay buffer).
# Finished runs are skipped; SKIP="sac_42 ..." additionally skips runs already in progress elsewhere.
P=${1:-2}
mkdir -p runs/logs
for s in 42 1 2; do for a in sac td3; do [[ " $SKIP " == *" ${a}_$s "* ]] || echo "$a $s"; done; done |
  xargs -P "$P" -L 1 bash -c 'python run_offpolicy.py --algo $0 --seed $1 > runs/logs/$0_s$1.log 2>&1'
