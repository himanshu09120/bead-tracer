#!/usr/bin/env bash
# Regenerate every table, figure and RESULTS.md from results/raw/*.json (no training, no policy rollouts).
# Add SLOW=1 to also re-run the slow-tracer simulation (writes results/slow_tracer.json first).
set -e
export PYTHONPATH=.
[ "${SLOW:-0}" = "1" ] && python -m analysis.slow_tracer
python -m analysis.main_table
python -m analysis.fair_speed
python -m analysis.metric_sensitivity
python -m analysis.mechanism
python -m analysis.figures
python -m analysis.results_md
