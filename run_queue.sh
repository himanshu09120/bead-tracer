#!/usr/bin/env bash
# Train every missing run, P at a time. Finished runs are skipped, interrupted ones resume.
# Existing seed-42 'full' run is NOT retrained (models/ + checkpoints/ are the reference).
P=${1:-4}
mkdir -p runs/logs
{
for s in 42 1 2 3 4; do echo "no_direction $s"; done
for s in 1 2 3 4;    do echo "full $s"; done
for s in 42 1 2;     do echo "no_direction_noisy $s"; done
for s in 42 1 2;     do echo "no_visual $s"; done
} | xargs -P "$P" -L 1 bash -c 'python run_experiment.py --obs-mode $0 --seed $1 > runs/logs/$0_s$1.log 2>&1'
