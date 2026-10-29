# SAC and TD3 baselines: experiments, results and paper changes

Status on 2026-10-30: **experiments finished and verified; manuscript NOT edited, PDF NOT regenerated.**
The repository contains no LaTeX source for the current paper (only the compiled PDFs `Himanshu_bead_tracer_paper__1_-5.pdf`
and `Himanshu_bead_tracer_paper__1_-7 (1).pdf`; the only `.tex` in git history is an early, different draft,
`paper/main.tex` in commit `c462f5e`), and this machine has no TeX installation. Section 4 therefore gives paste-ready LaTeX
rows and replacement text for each requested edit, written against the text of the `-7 (1)` PDF. Nothing was committed or pushed.

## 1. Files modified or created

Code (tracked files modified: 2, new: 4)

| File | Change |
|---|---|
| `evaluation/core.py` | `PPOPolicy(path, algo="ppo")` now loads `PPO`/`SAC`/`TD3` by name. Default behaviour is unchanged (PPO seed-42 path1 re-evaluated: 86.0348, identical to the stored result, max per-start difference 0). |
| `evaluate_model.py` | New options `--algo {ppo,sac,td3}` and `--method` (raw-file method name). Defaults reproduce the old behaviour. |
| `run_offpolicy.py` (new) | SAC/TD3 training, the off-policy counterpart of `run_experiment.py`. |
| `run_offpolicy_queue.sh` (new) | Trains the six runs, N at a time (`SKIP="sac_42"` skips runs started elsewhere). |
| `run_offpolicy_chains.sh` (new) | Two chains used to keep exactly two runs active after the seed-42 runs were started separately. |
| `run_offpolicy_eval.sh` (new) | Evaluates the six final models and runs the aggregation (see note in 5.4). |
| `analysis/sac_td3.py` (new) | Aggregates raw results into the Table VII/VIII/X rows. |

Artifacts (new)

| Path | Content |
|---|---|
| `models/{sac,td3}_s{42,1,2}_final.zip` | The six final models (copies; originals in `runs/<algo>_s<seed>/final.zip`, which is gitignored). |
| `models/{sac,td3}_s{42,1,2}_config.json` | Training configuration of each run (copy of `run.json`). |
| `results/raw/{sac,td3}_final__full__s{42,1,2}__<target>__test.json` | 78 raw evaluation files (6 models x 13 targets): per-start coverage, completion, return, steps, mean speed, end reason; trajectories for path1. |
| `results/sac_td3.json` | Per-seed and aggregate numbers. |
| `results/tables/sac_td3_{vii,viii,x}.tex` | Generated LaTeX rows (same content as section 4). |
| `runs/logs/{sac,td3}_s*.log`, `runs/logs/eval_*.log` | Training and evaluation logs. |
| `runs/logs/td3_diagnosis/`, `runs/td3_s42_FAILED_default_hparams/`, `runs/logs/td3_s42_FAILED_default_hparams.log` | Record of the failed first TD3 configuration (section 2.3). |
| `SAC_TD3_PAPER_CHANGES.md` | This file. |

PPO results, `config.py`, `env.py`, `train.py`, `run_experiment.py` and all existing raw results were not touched.

## 2. Training

### 2.1 What is identical to PPO
Environment (`light_env.EnvFn`, `BeadEnv`), target `target_images/path1.png`, `obs_mode="full"`, `config.ENV_KWARGS`
(rho = 8 px, phi = 0.95, 2500 max steps, stagnation limit 350, revisit penalty), curriculum spawn, 4 parallel env workers,
reward and coverage code, the `BeadFeaturesExtractor` CNN+MLP (one copy for the actor and one shared by the critic's two Q-networks, as SB3 builds them),
ReLU, 128-128 heads, 1,000,000 timesteps (exactly 1,000,000, since 1M is divisible by 4 envs), gamma = 0.995, learning rate 3e-4 for SAC,
continuous `Box(-1,1,(2,))` action space (SAC squashes with tanh, TD3 actor ends in tanh). Final models only; no checkpoint selection.

### 2.2 What differs (algorithm-specific, in `run_offpolicy.py`)

| Setting | SAC | TD3 | Why |
|---|---|---|---|
| Learning rate | 3e-4 | **1e-4** | TD3 changed after first configuration failed (2.3) |
| Replay buffer | 200,000 transitions | 200,000 | RAM: each transition stores two 64x64 crops; 15.5 GB machine |
| Learning starts / batch / tau | 10,000 / 256 / 0.005 | same | SB3 defaults |
| Updates | 1 gradient step per vec step (= 1 per 4 env transitions) | same | SB3 default; 1 per transition was ~4x slower (~25 h total) |
| Entropy | automatic temperature | - | SAC default |
| Exploration | stochastic policy | Gaussian action noise, sigma **0.3** | sigma 0.1 failed (2.3) |
| TD3 extras | - | policy delay 2, target noise 0.2 clipped at 0.5 | SB3 defaults |
| Q-network width | 128-128 (mirrors PPO's value head) | same | |
| Training device | CUDA (RTX 4060 Laptop) | CUDA | PPO was trained on CPU. GPU training is not bit-reproducible. |

The saved `run.json` files print `net_arch` from `config.POLICY_KWARGS` (`vf`) although `run_offpolicy.py` overrides it with `qf=[128,128]`;
`models/sac_s42_config.json` was written before the `--lr/--noise-sigma/--learning-starts` options existed and lacks those three fields (all `null` in the others).

Software: Python 3.13.9, PyTorch 2.13.0+cu130, Stable-Baselines3 2.9.0, Gymnasium 1.1.1, NumPy 2.5.2. Hardware: Intel Core Ultra 9 185H, 15.5 GB RAM, RTX 4060 Laptop GPU, Windows 11.

### 2.3 TD3 first configuration failed and was changed once (disclose in the paper)
With lr 3e-4 and sigma 0.1 (seed 42) the training coverage stayed at about 2% for 212k steps (episode length ~384 = stagnation limit,
return -59). A deterministic rollout of the 200k checkpoint showed the actor saturated at the corner action (+1, -1) with the bead pinned
at the arena wall (coverage 0.4%). The run was stopped and kept as evidence (`runs/td3_s42_FAILED_default_hparams`).
Two 100k-step variants were then run on seed 42 and compared on **training** coverage: sigma 0.3 (43.6%) and sigma 0.3 with lr 1e-4 (45.0%, return +32 vs -58);
the second was adopted for all three TD3 seeds (logs in `runs/logs/td3_diagnosis/`). The saturation check evaluated the variants
from 4 of the 20 test start points; those numbers were not used to choose between them. SAC and PPO were run once with their first configuration.
This asymmetry, and that the TD3 choice rests on one seed and 100k steps, is a limitation.

### 2.4 Seeds and commands
Seeds 42, 1, 2 for both algorithms (agent seed `s`, env seeds `s..s+3`). Wall time per run: SAC 2 h 43 m to 2 h 44 m, TD3 1 h 31 m to 1 h 39 m (two runs sharing the GPU).
```bash
# training (writes runs/<algo>_s<seed>/final.zip, run.json, done.json, tb/, ckpt/ [diagnostic only])
python run_offpolicy.py --algo sac --seed 42      # likewise sac 1, sac 2, td3 42, td3 1, td3 2
./run_offpolicy_queue.sh 2                        # all six, two at a time
# evaluation (CPU, 1 torch thread, deterministic, 20 test starts, 13 targets, phi = 0.95)
PYTHONPATH=. python evaluate_model.py --algo sac --method sac_final --label sac_s42 --seed 42 --train-phi 0.95 --model runs/sac_s42/final.zip
PYTHONPATH=. python evaluate_model.py --algo td3 --method td3_final --label td3_s42 --seed 42 --train-phi 0.95 --model runs/td3_s42/final.zip
python -m analysis.sac_td3                        # tables + results/sac_td3.json
```
(As actually run: seed-42 pair started with `run_offpolicy_queue.sh`, then `run_offpolicy_chains.sh` for seeds 1 and 2; each model was evaluated with the commands above.)

## 3. Results (all from `results/raw`, test starts, final models)

Per-seed path1 (mean +- SD over 20 starts, completed, mean return, mean steps, mean speed):

| Model | Coverage (%) | Completed | Return | Steps | Speed |
|---|---|---|---|---|---|
| SAC seed 42 | 78.7 +- 14.5 | 0/20 | 356.2 | 1725 | |
| SAC seed 1 | 85.2 +- 4.8 | 0/20 | 375.9 | 1582 | |
| SAC seed 2 | 88.0 +- 3.1 | 0/20 | 400.3 | 1442 | |
| **SAC, 3 seeds** | **84.0 +- 4.8** (SD of seed means) | **0/60** | 377.5 | 1583 | 3.62 |
| TD3 seed 42 | 85.7 +- 7.0 | 0/20 | 381.3 | 1653 | |
| TD3 seed 1 | 88.3 +- 2.6 | 0/20 | 389.7 | 1617 | |
| TD3 seed 2 | 86.7 +- 7.6 | 4/20 | 407.4 | 1911 | |
| **TD3, 3 seeds** | **86.9 +- 1.3** | **4/60** | 392.8 | 1727 | 3.76 |
| PPO, 5 seeds (existing, recomputed from raw) | 86.4 +- 4.7 | 5/100 | 421.4 | 1600 | 2.63 |
| Greedy (5 units/s) / pure pursuit (5 units/s) | 95.0 / 95.1 | 20/20 / 20/20 | 577.0 / 570.9 | 1817 / 1744 | 3.86 / 4.00 |

Cross-target (coverage %; held-out = mean over the ten shapes of per-shape mean, +- SD over shapes for one seed, +- SD of the three seed means for aggregates):

| Model | path1 | path2 | path3 | Held-out |
|---|---|---|---|---|
| SAC seed 42 | 78.7 | 51.9 | 51.9 | 69.8 +- 10.8 |
| SAC seed 1 | 85.2 | 44.5 | 47.5 | 82.7 +- 1.6 |
| SAC seed 2 | 88.0 | 53.9 | 53.5 | 85.0 +- 1.9 |
| SAC, 3 seeds | 84.0 +- 4.8 | 50.1 +- 5.0 | 51.0 +- 3.1 | 79.2 +- 8.2 |
| TD3 seed 42 | 85.7 | 52.9 | 54.3 | 77.4 +- 5.1 |
| TD3 seed 1 | 88.3 | 65.9 | 59.3 | 83.0 +- 7.1 |
| TD3 seed 2 | 86.7 | 56.2 | 47.1 | 77.0 +- 9.7 |
| TD3, 3 seeds | 86.9 +- 1.3 | 58.3 +- 6.7 | 53.6 +- 6.1 | 79.1 +- 3.4 |
| PPO, 5 seeds (Table IX, recomputed) | 86.4 +- 4.7 | 49.9 +- 8.0 | 49.3 +- 9.0 | 79.8 +- 6.4 |

Other facts used in the text: path1 end reasons over 60 episodes, SAC: 0 complete / 8 step limit / 52 stagnation; TD3: 4 / 8 / 48. Lowest and highest single-episode
coverage: SAC 52.6 / 94.7, TD3 69.6 / 95.1. Held-out completions: SAC 3 of 600 episodes, TD3 78 of 600 (77 of them from seed 2; seed 2 also has all 4 path1 completions).

## 4. Paper edits (paste-ready; source `.tex` not available)

### 4.1 Table VII (add after the PPO seed-42/2026 rows, before or after the five-seed rows)
Caption: change "PPO ROWS ARE FINAL MODELS UNLESS STATED" to "PPO, SAC AND TD3 ROWS ARE FINAL MODELS UNLESS STATED". Rows in the table's column order (Method & Coverage & Completed & Return & Steps):
```latex
SAC, seed 42, $\phi = 0.95$ & 78.7 $\pm$ 14.5 & 0/20 & 356.2 & 1725 \\
SAC, seed 1, $\phi = 0.95$ & 85.2 $\pm$ 4.8 & 0/20 & 375.9 & 1582 \\
SAC, seed 2, $\phi = 0.95$ & 88.0 $\pm$ 3.1 & 0/20 & 400.3 & 1442 \\
SAC, three seeds, $\phi = 0.95$ (SD of the three seed means) & 84.0 $\pm$ 4.8 & 0/60 & 377.5 & 1583 \\
TD3, seed 42, $\phi = 0.95$ & 85.7 $\pm$ 7.0 & 0/20 & 381.3 & 1653 \\
TD3, seed 1, $\phi = 0.95$ & 88.3 $\pm$ 2.6 & 0/20 & 389.7 & 1617 \\
TD3, seed 2, $\phi = 0.95$ & 86.7 $\pm$ 7.6 & 4/20 & 407.4 & 1911 \\
TD3, three seeds, $\phi = 0.95$ (SD of the three seed means) & 86.9 $\pm$ 1.3 & 4/60 & 392.8 & 1727 \\
```

### 4.2 Table VIII (after the PPO rows; keep the controller rows unchanged)
Caption: append "SAC AND TD3: FINAL MODELS, MEAN OF THE THREE SEED MEANS."
```latex
SAC full (final, 3 seeds) & 84.0 & 0/60 & 1583 & 3.62 \\
TD3 full (final, 3 seeds) & 86.9 & 4/60 & 1727 & 3.76 \\
```

### 4.3 Table X (after the PPO seed rows; the aggregate rows are optional if the table should stay per-seed only)
Caption: change "PPO ROWS ARE FINAL MODELS" to "PPO, SAC AND TD3 ROWS ARE FINAL MODELS"; for the aggregate rows add "THREE-SEED ROWS: MEAN $\pm$ SD OF SEED MEANS".
```latex
SAC trained on path1 (seed 42) & 78.7 & 51.9 & 51.9 & 69.8 $\pm$ 10.8 \\
SAC trained on path1 (seed 1) & 85.2 & 44.5 & 47.5 & 82.7 $\pm$ 1.6 \\
SAC trained on path1 (seed 2) & 88.0 & 53.9 & 53.5 & 85.0 $\pm$ 1.9 \\
SAC trained on path1 (3 seeds) & 84.0 $\pm$ 4.8 & 50.1 $\pm$ 5.0 & 51.0 $\pm$ 3.1 & 79.2 $\pm$ 8.2 \\
TD3 trained on path1 (seed 42) & 85.7 & 52.9 & 54.3 & 77.4 $\pm$ 5.1 \\
TD3 trained on path1 (seed 1) & 88.3 & 65.9 & 59.3 & 83.0 $\pm$ 7.1 \\
TD3 trained on path1 (seed 2) & 86.7 & 56.2 & 47.1 & 77.0 $\pm$ 9.7 \\
TD3 trained on path1 (3 seeds) & 86.9 $\pm$ 1.3 & 58.3 $\pm$ 6.7 & 53.6 $\pm$ 6.1 & 79.1 $\pm$ 3.4 \\
```

### 4.4 Section XIII-A, new paragraph (insert after the paragraph that ends "...do not trace faster.", before "The strongest learned policy...")
> To check whether the gap to the controllers is specific to PPO, we trained SAC and TD3 [27], [28] on path1 with the same environment, observation, reward, feature extractor and 1,000,000 timesteps, three seeds each (42, 1, 2), and took the final model of every run (Table VII; algorithm-specific settings are listed in Section X). Both learn to trace: the six final models cover 78.7% to 88.3% of path1 on average. SAC reaches 84.0 $\pm$ 4.8% (SD of the three seed means) and completes none of 60 starts; TD3 reaches 86.9 $\pm$ 1.3% and completes 4 of 60, all from seed 2. For comparison, the five PPO seeds give 86.4 $\pm$ 4.7% and 5 of 100. Their coverage ranges overlap, and with three seeds against five and no significance test we do not rank the three algorithms. None of them approaches the greedy controller (95.0%, 20 of 20 starts) or pure pursuit (95.1%, 20 of 20): 52 of the 60 SAC episodes and 48 of the 60 TD3 episodes on path1 end by the stagnation limit (88 of 100 for PPO), and the weakest single episodes cover 52.6% (SAC) and 69.6% (TD3). Across seeds, SAC's means span 78.7% to 88.0% and TD3's 85.7% to 88.3%, against 80.3% to 91.9% for PPO; the narrower TD3 range is an observation from three runs. SAC and TD3 move faster than PPO (3.62 and 3.76 versus 2.63 units/s, Table VIII), close to the tuned controllers (3.86 and 4.00), yet their episodes are about as long as PPO's (1583 and 1727 versus 1600 steps). Transfer follows the PPO pattern (Table X): on the open strokes SAC reaches 50.1% (path2) and 51.0% (path3) and TD3 58.3% and 53.6%, against 49.9% and 49.3% for PPO, while on the ten held-out shapes SAC and TD3 average 79.2% and 79.1% against 79.8% for PPO; the greedy controller scores about 95% on every target. Held-out completions are rare (3 of 600 episodes for SAC, 78 of 600 for TD3, 77 of them from seed 2). *Possible explanations, not tested:* that three different learning rules end in the same loop-and-truncate regime is consistent with the limit lying in the observation and task rather than in PPO's optimiser (Section XVI lists hypotheses), and TD3's replay and deterministic policy might explain its smaller spread; both would need more seeds and ablations to confirm. TD3 needed one change to its first configuration to learn at all (Section XVIII).

### 4.5 Section X (optional but needed for reproducibility; one paragraph where the PPO configuration is described)
> SAC and TD3 use the same environment, reward, observation and CNN--MLP feature extractor as PPO, with 128-128 actor and critic heads, $\gamma = 0.995$ and 1,000,000 timesteps over four environments. SAC uses learning rate $3\times10^{-4}$ and automatic entropy; TD3 uses learning rate $10^{-4}$, Gaussian exploration noise $\sigma = 0.3$, policy delay 2 and target-policy noise 0.2 (clipped at 0.5). Both use a replay buffer of 200,000 transitions (limited by memory), batch size 256, $\tau = 0.005$, 10,000 initial steps and one gradient update per four environment steps (the Stable-Baselines3 default for vectorised environments). They were trained on a GPU; evaluation is identical to PPO's (CPU, one thread, same 20 test starts).

New references (append): `[27] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, "Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor," in Proc. International Conference on Machine Learning (ICML), PMLR, vol. 80, 2018, pp. 1861--1870.`
`[28] S. Fujimoto, H. van Hoof, and D. Meger, "Addressing function approximation error in actor-critic methods," in Proc. International Conference on Machine Learning (ICML), PMLR, vol. 80, 2018, pp. 1587--1596.`

### 4.6 Section XVIII, Limitations and Future Work
Replace the sentence "The main comparison uses five seeds for the full and no-direction tiers, three seeds for the noisy and no-visual tiers, and each reward-ablation cell two." with:
> The main comparison uses five seeds for the full and no-direction tiers, three seeds for the noisy and no-visual tiers and for SAC and TD3, and each reward-ablation cell two.

Replace "PPO hyperparameters were not tuned beyond the original configuration, and other algorithms such as SAC and TD3 were not compared." with:
> Hyperparameters were not tuned systematically for PPO, SAC or TD3. SAC and TD3 were run with three seeds each, a replay buffer of 200,000 transitions (limited by memory) and one gradient update per four environment steps; the comparison is therefore between one configuration of each algorithm, not between tuned algorithms. TD3's first configuration (learning rate $3\times10^{-4}$, exploration noise 0.1) collapsed to a saturated corner action in seed 42; it was changed once, on the basis of training coverage, to the values in Section X for all three seeds, whereas PPO and SAC used their first configuration. Other off-policy and on-policy algorithms, and SAC and TD3 trained on a bank of shapes, were not tried.

In "The most useful next steps are:" replace "comparing SAC and TD3 under the same protocol;" with "tuning SAC and TD3 and testing further algorithms under the same protocol;". Retained unchanged: simulation-only, single-run phi = 0.80 results, 64-shape model, one family of smooth closed curves, fixed rho, untested explanations of the image effect.

Other places that mention an algorithm comparison and need no change: Abstract, Section XVII ("no learned policy exceeds that law"; still true: best SAC/TD3 seed mean is 88.3% against 95.0%) and Section XIX (conclusion). If desired, add "SAC and TD3 behave like PPO" to the Section XVII "Learning behaviour" paragraph.

## 5. Validation

1. **Raw vs summaries:** for all 78 files, `coverage_mean/sd`, `n_complete`, `ret_mean`, `steps_mean`, `speed_mean` recomputed from the 20 per-start episodes: 0 mismatches. Every file has 20 test starts, split `test`, phi 0.95, obs `full`; the 20 start points of all six models equal those of the PPO path1 raw results.
2. **Regeneration:** re-evaluating `runs/sac_s2/final.zip` on path1 and `runs/td3_s1/final.zip` on heldout03 with the existing harness reproduced the stored per-start coverage exactly (max difference 0). Training itself is not bit-reproducible (GPU).
3. **PPO unaffected:** `models/bead_ppo_final.zip` path1 re-evaluated through the edited `PPOPolicy`: 86.0348 = stored value; `pytest tests -m "not slow"`: 2 passed (slow reproducibility tests not rerun). PPO five-seed reference numbers (86.4 +- 4.7, 5/100, return 421.4, steps 1600, speed 2.63; path2 49.9, path3 49.3, held-out 79.8 +- 6.4) and the greedy/pure-pursuit rows recompute from `results/raw` and match the PDF.
4. **Aggregation:** `analysis/sac_td3.py` conventions: coverage SD over starts uses population SD like the existing PPO rows; seed-mean SD and held-out SD use ddof = 1; completions are summed; return/steps/speed are means of seed means. `run_offpolicy_eval.sh` was written to bundle the evaluation commands but was not itself executed end to end (the same commands were run individually).
5. **Stored `seed` field** in the new raw files is the string `"42"` (as `evaluate_model.py` writes it), whereas `evaluate_run.py` stores an int; `analysis/sac_td3.py` does not depend on it.
6. **Compilation:** not possible (no `.tex`, no TeX toolchain). Table consistency checked on the plain numbers above only; cross-references in 4.4 (Tables VII, VIII, X, Sections X, XVI, XVIII) must be checked after pasting.

### Unresolved
- Manuscript source is missing: edits in section 4 are not applied and the PDF is not regenerated. Put the Overleaf/`.tex` sources in the repo (or paste the section 4 text into Overleaf) and recompile.
- Hyperparameter asymmetry (TD3 adjusted once, SAC/PPO not), three seeds only, one 200k replay buffer and a 0.25 update-to-data ratio.
- GPU-trained models: retraining with the same seed gives similar but not identical weights.
- New citations [27], [28] were typed from memory of the standard references; check them against the bibliography before submission.
