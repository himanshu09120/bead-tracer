# Professor feedback changelog

Written for: another researcher or AI continuing this revision without repeating the audit.

## 1. Executive summary

| # | Professor comment | Status |
|---|---|---|
| 1 | Re-run phi = 0.80 with a recorded seed; update Tables VII and X | **Experiment done** (seed 2026, 1,003,520 steps, evaluated). **Manuscript NOT edited**: no LaTeX source exists (see section 2). Paste-ready rows are in section 5. |
| 2 | Fill missing held-out entries in Table X (phi = 0.80 model, 64-shape model) | phi = 0.80: **done for both models** (historical unrecorded-seed model 85.2 +- 1.2; new seed-2026 model 90.9 +- 0.9). 64-shape model: **unavailable**, checkpoint and training script are not in the repo or on this machine (section 11). |
| 3 | Strengthen the no-visual explanation | Existing numbers **verified** against raw data. Paragraph for Section XVI is written in section 7 and still needs inserting into the manuscript. |

Nothing was committed or pushed. No LaTeX was compiled: there is no source and no TeX installation on this machine.

## 2. Repository baseline

- Repo: https://github.com/himanshu09120/bead-tracer.git, branch `test1`, HEAD `7270bb72deb9051668e2ce82ecf6e0518ad097af` ("Add files via upload"). Working tree before my changes: untracked `benchmark/`, `runs_baselines.log` (left untouched).
- **Manuscript:** the only copy of the current paper in the repo is the PDF `Himanshu_bead_tracer_paper__1_-5.pdf` (added in HEAD). There is **no `.tex` or `.bib`** for it. `git log --all` shows only one earlier paper source, `paper/main.tex` and `paper/bead_tracer_paper_overleaf.zip` in commit `c462f5e` (branch `origin/test`). It is a different, early draft (title "Bead Curve Reconstruction: ...", 82.03% result), not the current paper. I did not use or recreate it. A disk search found no other copy.
- Because the PDF is the only manuscript, there is nothing to diff against it; no reference-PDF differences exist to report.
- Code and artifacts: `config.py` (hyperparameters), `env.py` (environment, `obs_mode`), `train.py`, `run_experiment.py` (seeded training), `evaluation/core.py` and `evaluation/runs.py` (protocol), `evaluate_run.py`, `analysis/*.py`, `results/raw/*.json`, `results/tables/*.tex`, `shapes/` (generator, `train/` 100 shapes, `heldout/` 10 shapes), `models/`, `runs/`.

### Configuration audit (paper vs code)

All of these match the paper: learning rate 3e-4, n_steps 1024 x 4 envs, batch 256, 10 epochs, gamma 0.995, GAE lambda 0.95, clip 0.2, vf 0.5, ent 0.005, grad clip 0.5, 1,000,000 timesteps (runs end at 1,003,520 after the last full rollout), arena 10 (`env.ARENA`), max speed 15 (`MAX_SPEED`), rho = 8 px, 2500 max steps, 350 stagnation limit, 64x64 crop, 11-dim state, 20 test starts without jitter (`env.eval_start_points`), PPO on CPU, 1 torch thread.

Seed propagation: `build_model(seed=)` seeds Python, NumPy and torch through SB3; each env is seeded with `seed + rank` in `light_env.EnvFn`.

Discrepancies found (not resolved by me):
1. Paper says "64 procedural shapes"; `shapes/train/` has 100 shapes (seeds 1000-1099). No script or checkpoint for the 64-shape run exists.
2. Historical phi = 0.80 model on path1: paper Table VII says 94.8 +- 0.3, 12/20, 474.5, 823. Re-evaluating the same file in the current harness gives 94.86 +- 0.33, 13/20, 480.4, 803 (`tests/reference_values.json`, `RESULTS.md`; `tests/test_repro.py` expects 12). README documents that rollouts are chaotic and differ slightly across machines.
3. The 163258 model is not in the README-stated paper tables, yet the paper's Table VII and X list it as "seed unrecorded". README says the seed is unknown.
4. Table XIII caption calls the curvature columns "percentage of episodes in which a pixel of the class was left uncovered"; `analysis/mechanism.py` computes a per-class miss rate with bootstrap CIs. Worth checking the wording.

## 3. Files modified or created

Nothing tracked was modified. New (all untracked, nothing committed):
- `evaluate_model.py`: evaluates one standalone model file on test starts of all 13 targets with the existing harness (final model only, no checkpoint selection).
- `analysis/table_x_extra.py`: aggregates the two phi = 0.80 models into Table VII and X rows; writes `results/table_x_extra.json`.
- `results/table_x_extra.json`: aggregated numbers, per-start and per-shape values.
- `results/raw/ppo_m163258__full__sunrecorded__*__test.json` (13 files): historical model.
- `results/raw/ppo_phi080_s2026__full__s2026__*__test.json` (13 files): new model.
- `runs/phi_080_seed_2026/` (gitignored): `final.zip`, `checkpoints/` (every 50k), `run.json` (config), `done.json`, `tb/`. Log: `runs/logs/phi_080_seed_2026.log`.
- `PROFESSOR_FEEDBACK_CHANGELOG.md` (this file).

## 4. Reproducibility record (new run)

- Seed **2026** (PPO seed and env seeds 2026-2029). Commit `7270bb7`. Run date 2026-10-25.
- Hardware: Intel Core Ultra 9 185H, 15.5 GB RAM, Windows 11; training on CPU (2 torch threads, 4 env workers); an RTX 4060 is present but unused. Training wall time 3541 s.
- Software: Python 3.13.9, PyTorch 2.13.0+cu130, Stable-Baselines3 2.9.0, Gymnasium 1.1.1, OpenCV 5.0.0, NumPy 2.5.2.
- Configuration: unchanged `config.py` values; only `completion_frac` = 0.80 (`runs/phi_080_seed_2026/run.json`). Target `target_images/path1.png`, obs mode `full`.
- Evaluation: final model, CPU, 1 torch thread, deterministic, phi = 0.95, rho = 8 px, the 20 test starts without jitter (`env.reset(seed=1000+i)`). No checkpoint selection, no use of test results to choose anything.
- Commands:
```bash
python run_experiment.py --obs-mode full --seed 2026 --phi 0.80 --name phi_080_seed_2026
PYTHONPATH=. python evaluate_model.py --model runs/phi_080_seed_2026/final.zip --label phi080_s2026 --seed 2026 --train-phi 0.80
PYTHONPATH=. python evaluate_model.py --model models/bead_ppo_20260819_163258_final.zip --label m163258 --seed unrecorded --train-phi 0.80
PYTHONPATH=. python -m analysis.table_x_extra
```

## 5. Before and after results

Table VII row (path1, 20 test starts, phi = 0.95). Sources: `results/raw/ppo_phi080_s2026__full__s2026__path1__test.json`, `results/raw/ppo_m163258__full__sunrecorded__path1__test.json`.

| Run | Coverage (%) | Completed | Return | Steps |
|---|---|---|---|---|
| Old paper row (seed unrecorded) | 94.8 +- 0.3 | 12/20 | 474.5 | 823 |
| Same file, re-evaluated here | 94.9 +- 0.3 | 13/20 | 480.4 | 803 |
| **New, seed 2026** | **94.3 +- 1.1** | **9/20** | **462.0** | **915** |

Seed 2026 per-start ends: 9 complete, 11 stagnation, none at max steps. The new run is close to, but not equal to, the historical one: coverage 0.5 pp lower, 3 to 4 fewer completions, more steps. I did not investigate this beyond the numbers; with one run each, the difference cannot be attributed to anything (seed, machine, chaotic rollouts).

Table X, phi = 0.80 rows, 20 test starts, evaluated at phi = 0.95. Held-out = mean over 10 shapes of the per-shape mean, +- sample SD (ddof = 1) over the 10 shapes (`analysis/table_x_extra.py`).

| Model | path1 | path2 | path3 | Held-out shapes |
|---|---|---|---|---|
| Historical, seed unrecorded | 94.9 | 57.8 | 63.8 | **85.2 +- 1.2** |
| **New, seed 2026** | **94.3** | **55.4** | **61.2** | **90.9 +- 0.9** |
| 64 procedural shapes | 53.8 | 58.6 | 69.3 | **unavailable** (section 11) |

Per-shape means, historical: 85.46, 86.19, 85.25, 83.14, 87.18, 84.19, 85.75, 84.00, 86.26, 84.34. New: 91.47, 90.83, 90.17, 91.26, 91.62, 89.59, 92.10, 91.89, 90.88, 89.65. Pooling all 200 episodes gives the same means here (because every shape has 20 starts), but the reported figure is the shape-level mean.

Held-out shapes were not used for training: `shapes/heldout/` (seeds 9000-9009) is disjoint from `shapes/train/` (1000-1099), and both phi = 0.80 models were trained on `target_images/path1.png` only. The historical model was trained on 2026-08-19, before the shapes existed (2026-10-21).

Suggested manuscript edits (paste into the source):
- Table VII: replace the "seed unrecorded" row with "PPO, seed 2026, trained with phi = 0.80 | 94.3 +- 1.1 | 9/20 | 462.0 | 915". Keep the old numbers as a stated historical result (text or footnote), since the old run still exists and is evaluated with a different completion count.
- Table X: row "PPO trained on path1 (seed 2026, phi = 0.80) | 94.3 | 55.4 | 61.2 | 90.9 +- 0.9". If the historical row is kept, fill it with 85.2 +- 1.2. Leave the 64-shape cell as the dash and keep the caption saying it was not evaluated.
- Section XIV, replacement for the sentences about phi = 0.80 and the 64-shape bank: "The phi = 0.80 model of Table VII (seed 2026) scores 55.4% on path2 and 61.2% on path3 when evaluated at phi = 0.95, and 90.9 +- 0.9% on the ten held-out shapes; the earlier run with an unrecorded seed scores 57.8%, 63.8% and 85.2 +- 1.2%. Both are single runs, so the difference between them is not interpretable. The model trained on a bank of 64 procedural shapes could not be re-evaluated on the current held-out shapes because its checkpoint is not available, so the path1-only comparison with it is limited to the earlier path1, path2 and path3 results."
- Discussion/abstract/conclusion: change "(seed unrecorded) reaches 94.8% and completes 12 of 20" to the seed-2026 numbers, keep "this result rests on one run", and keep "five phi = 0.95 runs reach lower coverage on average" only as a description, not a claim that phi = 0.80 is better. The held-out comparison with the phi = 0.95 seeds (70.9 to 87.0) is also a single run against five, and should not be presented as a superiority claim. Also search for other mentions of "unrecorded" (Section XIV, limitations, Section XVII).

## 6. Raw data and calculations

- Per-start coverage, completion, return, steps, end reason, mean speed: `episodes` list in each `results/raw/ppo_*__test.json`. `summary` gives mean, SD (population, as in the existing harness), completions, mean return, mean steps.
- Held-out aggregation: `analysis/table_x_extra.py`, mean of ten per-shape `coverage_mean` values, SD with `ddof=1`.
- Table IX and XIII verification (section 8) was recomputed directly from `results/raw/ppo_final__<tier>__s<seed>__*__test.json`.

## 7. Scientific explanation (paragraph for Section XVI)

Measured facts, from raw data (final models, path1, mean over seed means):
- Coverage: full 86.4 +- 4.7% (5 seeds) vs no-visual 94.1 +- 0.8% (3 seeds). Seed sets are not matched and no significance test was run.
- Share of missed pixels that belong to stagnation-terminated episodes: 93.0% (full) vs 19.4% (no-visual). Episode level, same data: 88/100 full episodes end in stagnation vs 9/60 for no-visual; no-visual instead has 39/60 completions and 12/60 ending at the step limit. Mean steps 1600 vs 2252 (no-visual episodes last longer because they keep progressing).
- Mean speed is the same: 2.63 vs 2.63 world units/s, so no-visual is not simply faster or slower.
- Miss rate by curvature class is lower in every class (full 14.4 / 11.7 / 13.7%; no-visual 5.8 / 5.5 / 6.0%), so the gain is not specific to concave regions.
- Code check: `obs_mode="no_visual"` only zeroes the 64x64 crop (`env.py`, `_get_obs`); the 11-dim state, including the unit direction to the nearest uncovered pixel and its distance (`state[4:7]`), is unchanged.

Paragraph:

> The no-visual tier reaches 94.1% mean coverage on path1, compared with 86.4% for the full observation, and 39 of its 60 test episodes complete the task against 5 of 100. Removing the image also changes how episodes end: 93.0% of the target pixels missed by the full-observation policies belong to episodes ended by the stagnation limit, against 19.4% for the no-visual policies, and 88 of 100 full-observation episodes end by stagnation against 9 of 60 for no-visual. Mean speeds are the same (2.63 world units/s in both tiers), and the lower miss rate holds in all three curvature classes. Importantly, the no-visual agent still receives the numerical direction toward the nearest uncovered contour pixel and its distance; it removes only the local crop. These measurements establish that the image is associated with more stagnation-terminated episodes, but the ablation does not establish why. We consider the following explanations as hypotheses that have not been tested. First, the direction features may already carry the information required for navigation, so the crop is redundant and may add conflicting signals. Second, the 64 x 64 crop, about 1.6 world units across, may expose nearby contour fragments that attract the agent into repeated local traversal, and it offers no global context to tell a productive move from a loop; the loop-and-truncate behaviour described above would follow. Third, the local coverage reward combined with the 350-step stagnation limit may allow temporary progress before a loop, which a policy driven by a low-dimensional feedback law could avoid. Fourth, the convolutional branch may make optimisation harder or compete with the numerical branch; we did not measure gradients or training dynamics and make no claim about this. These comparisons also use different seed sets (five versus three), and no significance test was performed. Resolving the mechanism requires matched seeds and identical budgets, several independent seeds per tier, logging of stagnation and coverage during training, gradient analysis where feasible, and controlled changes to the crop size or wider visual context.

## 8. Claim verification

- Table IX: **all 4 tiers reproduced exactly** (86.4 +- 4.7, 5/100, 49.9, 49.3, 79.8 +- 6.4; 86.4 +- 8.2, 8/100; 63.3 +- 21.0, 0/60; 94.1 +- 0.8, 39/60; 91.8; 90.2; 94.9 +- 0.2).
- Table XIII (`results/tables/mechanism.tex`): stagnation shares 93.0 / 94.9 / 100.0 / 19.4 and curvature miss rates **match** the paper. The 2.6-2.8 speed range holds for full (2.63), no-direction (2.79) and no-visual (2.63); the noisy tier is lower (1.85, the paper's 1.93 for Table XIII), so "relevant tiers" excludes it.
- Qualified: phi = 0.80 versus phi = 0.95. One run cannot support superiority; the paper's own explanation (lower threshold ends episodes before the loop regime) remains a hypothesis.
- Retained but flagged: the "efficient/matches greedy" statements were not re-examined.
- Unresolved: any claim about the 64-shape model (section 11).

## 9. Validation

- Training completed (1,003,520 steps, no errors in `runs/logs/phi_080_seed_2026.log`).
- Both evaluations completed for all 13 targets; counts and ranges checked (each result has 20 episodes).
- Table IX and XIII recomputed from raw files and match.
- pytest was not re-run. LaTeX compilation: **not performed** (no source, no TeX installation).

## 10. Reproduction instructions

```bash
pip install -r requirements.txt
python run_experiment.py --obs-mode full --seed 2026 --phi 0.80 --name phi_080_seed_2026
PYTHONPATH=. python evaluate_model.py --model runs/phi_080_seed_2026/final.zip --label phi080_s2026 --seed 2026 --train-phi 0.80
PYTHONPATH=. python evaluate_model.py --model models/bead_ppo_20260819_163258_final.zip --label m163258 --seed unrecorded --train-phi 0.80
PYTHONPATH=. python -m analysis.table_x_extra       # prints Table VII/X rows, writes results/table_x_extra.json
./run_all.sh                                        # existing tables 2-13 (unchanged)
```
Paper compilation: no build workflow exists in this repo.

## 11. Remaining work

1. **Manuscript edits** are not done, because the LaTeX source is not in the repo. Next action: obtain the Overleaf/LaTeX source (the 2026-10-25 PDF), apply section 5 edits to Tables VII and X and Section XIV, insert the section 7 paragraph in Section XVI, search for "unrecorded", then compile.
2. **64-shape model, Table X held-out cell:** no checkpoint, no multi-shape training script, and the paper's 64 does not match the 100 shapes in `shapes/train/`. Next action: locate the original model/script (ask the author); evaluate with `evaluate_model.py` using `--label shapes64 --seed <seed>`. Reproducing it would need an undocumented design (which 64 shapes, one environment per shape or random shape per episode); that would be a new model and must be labelled as such.
3. Investigating why seed 2026 completes 9/20 vs 12-13/20 for the historical run needs more seeds; I did not run any.
4. Git: nothing is committed. If you commit, use identity himanshu09120 / hsinha0912@gmail.com (already set locally) and no co-author line.
