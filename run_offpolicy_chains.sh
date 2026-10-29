#!/usr/bin/env bash
# Two chains, never more than 2 runs at once (each holds a ~1.7 GB replay buffer):
#   chain 1 waits for td3_s42 to finish, then trains sac_s1, td3_s1
#   chain 2 waits for sac_s42 to finish, then trains sac_s2, td3_s2
# (used when runs/{sac,td3}_s42 were started separately; run_offpolicy_queue.sh is the one-shot equivalent)
run() { python run_offpolicy.py --algo $1 --seed $2 > runs/logs/$1_s$2.log 2>&1; }
wait_for() { until [ -f runs/$1/done.json ]; do sleep 20; done; }
mkdir -p runs/logs
( wait_for td3_s42; run sac 1; run td3 1 ) &
( wait_for sac_s42; run sac 2; run td3 2 ) &
wait
