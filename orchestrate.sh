#!/usr/bin/env bash
# Unattended finish: staged eval workers -> analyses -> tests -> commit.  Log: runs/logs/orchestrate.log
export PYTHONPATH=.
cd "$(dirname "$0")"
log() { echo "$(date +%H:%M:%S) $*" >> runs/logs/orchestrate.log; }
# the legacy sequential queue is replaced by locked workers; its in-flight evaluation (no_direction_s42) is left to finish
mkdir runs/no_direction_s42/.lock 2>/dev/null
while pgrep -f "evaluate_run.py --run no_direction_s42" > /dev/null 2>&1 || powershell -c "if (Get-CimInstance Win32_Process | ? { \$_.CommandLine -match 'evaluate_run.py --run no_direction_s42' -and \$_.CommandLine -notmatch 'Get-Cim' }) { exit 0 } else { exit 1 }"; do sleep 30; done
touch runs/no_direction_s42/.eval_done; log "s42 eval finished; starting worker A"
./eval_worker.sh & 
until [ $(ls runs/*/final.zip 2>/dev/null | wc -l) -ge 15 ]; do sleep 120; done
log "training finished; starting workers B C D"
./eval_worker.sh & sleep 20; ./eval_worker.sh & sleep 20; ./eval_worker.sh &
wait
log "all evaluations done; running analyses"
./run_all.sh >> runs/logs/orchestrate.log 2>&1 && log "run_all ok" || log "run_all FAILED"
python -m pytest tests -q >> runs/logs/orchestrate.log 2>&1 && log "pytest ok" || log "pytest FAILED"
git add results RESULTS.md >> runs/logs/orchestrate.log 2>&1
git commit -qm "Add learned-method results, tables, figures and RESULTS.md" >> runs/logs/orchestrate.log 2>&1 && log "committed"
log "ALL DONE"
