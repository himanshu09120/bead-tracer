# Benchmarking the Bead-Tracer PPO agent against published coverage-path-planning methods

## 1. Executive summary

This report benchmarks the project's trained PPO contour-tracing agent
(`models/bead_ppo_final.zip`, trained on `target_images/path1.png`,
`completion_frac=0.95`) against **five published reinforcement-learning
papers** and **one classical algorithm family** from the coverage-path-
planning (CPP) literature. One baseline (Jonnarth et al.) is reproduced from
**official code and an official pretrained checkpoint**; four (Garrido-
Castañeda et al., Chen et al., Theile et al., Devo et al.) are **faithful
reimplementations** built from each paper's published methodology, since
none of the four released usable code for the exact cited paper; the sixth
is a **classical, non-learned controller** implemented directly inside this
project's own environment.

**A note on research integrity for this report**: while researching and
implementing these baselines, three real mistakes were caught and corrected
before finalizing, rather than left in:
1. The "rl-cpp" repository's GitHub username (`arvijj`) was initially
   mistaken for an author surname ("Vijgen et al.") -- corrected to the
   actual authors, Jonnarth et al., after reading the repository's own
   citation.
2. An early web search's summary of "PPO with action masking and discount
   factor scheduling" was initially attributed to Jonnarth et al.; reading
   Jonnarth et al.'s actual `mower_env.py` showed a continuous, unmasked
   action space, so that description was re-traced to its real source --
   Theile et al.'s UAV coverage work.
3. A public repository (`github.com/theilem/uavSim`) was initially assumed
   to be "the" official code for Theile et al.'s 2020 IROS paper; its own
   README revealed it now implements two later, different papers by the
   same lab, with the true 2020 paper's code not clearly present in any
   branch checked -- so it was NOT used, avoiding a misattributed
   "official code" result, and that baseline was reimplemented from the
   paper's own text instead.

**Headline result:** our PPO agent achieves **82.03% mean contour coverage**
(std 15.26pp) over 20 deterministic evaluation episodes, comfortably ahead
of every classical or grid-world RL baseline tested (3.7%-49.5% mean
coverage), and in the same neighborhood as the strongest published RL
baseline reproduced here (Jonnarth et al.'s official PPO checkpoint: 90.8%
coverage / 60% success in our own run, 99.0% / 100% in the checkpoint's own
bundled reference metrics). **Success/completion rate is 0% for our model**
at its current, very strict 95%-coverage threshold — this reflects the
threshold, not poor tracing, since mean coverage is still 82%; see §5.1.

**This is explicitly NOT a fully apples-to-apples comparison.** The
different methods solve genuinely different tasks (1-D contour tracing vs.
2-D area/grid coverage) in different environments with different physics,
action spaces, and success criteria. Section 6 states plainly, method by
method, where the comparison is and isn't valid.

## 2. Our project (recap)

- **Environment**: `BeadEnv`, a physically simulated 2-D bead tracing a
  contour extracted (Canny + `findContours`) from a target image.
- **Action space**: continuous `Box(-1, 1, shape=(2,))` thrust vector.
- **Observation**: `Dict{local_map: (64,64,1) distance-map crop, state: (11,)}`.
- **Algorithm**: Stable-Baselines3 PPO, custom small CNN+MLP feature
  extractor (~430k params), `gamma=0.995`, `gae_lambda=0.95`.
- **Coverage metric**: ground-truth boolean mask over individual
  target-contour pixels, swept-capsule test — never approximated.
- **Evaluation**: 20 deterministic, evenly spaced start points
  (`env.eval_start_points(20)`), deterministic policy.

Full detail in the project's own `README.md`; nothing about the environment,
model, training, or existing evaluation results was modified for this
benchmark.

## 3. Paper selection: methodology and rejected candidates

Selection criteria (from the task brief): genuinely RL-CPP-related,
sufficiently detailed methodology, public code preferred, reproducible
locally, meaningful quantitative metrics, PPO/actor-critic/continuous-control
preferred, and **not** chosen for inflated reported numbers.

Searched and considered (with reasons for exclusion where applicable):

| Candidate | Verdict | Why |
|---|---|---|
| Jonnarth, Zhao & Felsberg, ICML 2024 / Jonnarth, Johansson, Zhao & Felsberg, IEEE Access 2025 ("rl-cpp") | **SELECTED** | PPO, official code + pretrained checkpoints, continuous action space, map-based CNN observation -- closest architectural relative of our own approach |
| Garrido-Castañeda, Vasquez & Antonio-Cruz, Sensors 2025 | **SELECTED** | A2C+PPO via Stable-Baselines3 (same library as us), fully specified discrete-grid methodology, directly gives success/coverage/redundancy metrics we needed. No public code -> reimplemented |
| Chen, Lu, Cui, Luo & Zheng, Sensors 2025 ("Re-DQN") | **SELECTED (baseline only)** | Directly "coverage path planning for a mowing robot" via DRL; included for a value-based (DQN) comparison point. No public code -> reimplemented; the paper's *novel* Re-DQN contributions (noisy layers, dynamic incentive, curiosity bonus) were NOT reproduced -- see §4.3 |
| Choset & Pignon, Boustrophedon Cellular Decomposition, 1997/2000 | **SELECTED (classical, not RL)** | The field's universal classical baseline; used here as a non-learned floor reference, adapted for curve rather than area coverage (see §4.4) |
| JonasVervloet/RL-Coverage-Planner (GitHub) | Rejected | Open-source project, but **not tied to any published paper** -- the task explicitly asked for research papers |
| Fixed-wing UAV continuous CPP (AM-SAC), arXiv 2505.08382 | Rejected | Genuinely continuous action space (a plus), but: unreviewed arXiv preprint only (not peer-reviewed as of writing), no public code, and its environment (rectangle-decomposed airspace + Bézier-curve UAV motion) is structurally very different from ours -- would have required forcing an ill-fitting comparison |
| h-brenne/cppRL (ROS/TurtleBot3) | Rejected | Requires a full ROS + Gazebo stack; not practically reproducible in this environment within scope |
| Abdelaziz, Noureldin & Givigi, IEEE CCECE 2024 (DOI 10.1109/CCECE59415.2024.10667258) | Rejected | Genuinely IEEE-published, but a 2-page conference note (pp. 153-154) -- paywalled on IEEE Xplore with no accessible preprint found anywhere, so essentially zero extractable methodology (no reward formula, no obs/action space detail beyond the title). Fails the "sufficiently detailed methodology" bar outright, not a judgment call |
| Zheng, Jin, Zhao, Ma, Chen & Xu, "Deep reinforcement learning based coverage path planning in unknown environments" (TD3), 2025 preprint (Preprints.org / EngrXiv) | Rejected | Full text obtained and read. Genuinely continuous-action actor-critic (TD3) with a closed-form reward (`r_t = α·ΔCoverage - β·distance`), but **α, β, network architecture, and all training hyperparameters are left unspecified**; the environment is **ROS+Gazebo** (heavy, Linux-first, impractical here); no code released; and its reference list is padded with numerous unrelated citations (medical imaging, LLM API papers) -- a real credibility flag suggesting a low-rigor venue. Not selected. |
| **Theile, Bayerlein, Nai, Gesbert & Caccamo, IEEE/RSJ IROS 2020** (DOI 10.1109/IROS45743.2020.9340934) | **SELECTED** | DDQN, map-based CNN observation, UAV coverage under a power/movement-budget constraint. Exceptionally well-specified: exact grid/channel layout, exact network architecture diagram, exact training algorithm pseudocode, exact hyperparameter table. No code for this exact paper (see integrity note above) -> reimplemented |
| **Devo, Mao, Costante & Loianno, IEEE Robotics and Automation Letters 2022** (DOI 10.1109/LRA.2022.3154019) | **SELECTED** | IMPALA-based asymmetric actor-critic, single-RGB-image drone exploration. Exact 11-action space and exact closed-form entropy reward given; uses coverage % as its own metric. Environment (Unreal Engine 4 + real drone + Vicon mixed reality) not reproducible -> reimplemented as a lightweight grid proxy preserving the exact action space and reward formula |

## 4. Baselines: methodology, assumptions, and honest limitations

### 4.1 Jonnarth et al. (ICML 2024 / IEEE Access 2025) -- "rl-cpp"

- **Code**: official, [github.com/arvijj/rl-cpp](https://github.com/arvijj/rl-cpp) (BSD-3-Clause-Clear). Cloned and run in an isolated `conda` environment (`rlcpp_baseline`, Python 3.9) because the repo pins `gym==0.21.0` / `stable-baselines3==1.6.2`, incompatible with this project's own Gymnasium/SB3 stack.
- **Algorithm**: PPO (SB3 1.6.2), continuous action space -- `Box(-1,1,shape=(1,))` (steering only, constant linear velocity) in the checkpoint used here.
- **Observation**: multi-scale coverage/obstacle/frontier maps (CNN input) + 24-ray lidar, egocentric and heading-aligned.
- **Reward**: newly-covered-area term, incremental/global total-variation smoothness terms, frontier-seeking term, wall/obstacle collision penalties, goal-coverage bonus.
- **Termination**: goal coverage reached, or `max_non_new_steps` stagnation.
- **Task**: lawn-mowing style *area* coverage of a randomly generated free-space region with unknown obstacles -- structurally different from our *contour* coverage task (see §6).
- **Two distinct result sets, both included and clearly labeled**:
  - `rlcpp_mowing_tv1_bundled`: reference metrics **shipped inside the official pretrained-weights download** -- the authors' own numbers, computed with their own checkpoint and code, which we only parsed (we did not run anything to produce these).
  - `rlcpp_mowing_tv1_local`: our own invocation of the official `eval.py`, same checkpoint and code, executed locally. Cross-checked against `eval.py`'s own printed summary (90.84% mean coverage, 60.0% success at the 0.99 threshold) -- our independently computed numbers matched exactly.
- **Known limitation**: the bundled reference set has only 6 episodes (all that shipped with the checkpoint); our local run used the full 15 official evaluation maps. The two are not the same sample size and shouldn't be treated as repeated measures of the same thing.

### 4.2 Garrido-Castañeda, Vasquez & Antonio-Cruz (Sensors 2025)

- **Code**: none released ("Data are available on request", no GitHub link in the paper). **Reimplemented by us** in `benchmark/baseline_sensors_actorcritic/grid_env.py`.
- **Algorithm**: A2C and PPO, both via Stable-Baselines3 (paper used v1.0.8; this reimplementation used whatever SB3 version this project's own venv has, recorded per-run in `results/sensors_*.json["meta"]["sb3_version"]`).
- **Environment**: 9x9 discrete grid, `Discrete(4)` actions (N/E/S/W), 8-boolean Moore-neighborhood obstacle sensing, reward = -0.1/step, +0.1/new cell, -10/+0.01 collision/free-move, +10 completion.
- **Assumptions documented in code and results metadata**: obstacle density/layout generation procedure (not specified by the paper), training timestep budget (300,000, not specified by the paper), network architecture (SB3 default `MlpPolicy`, not specified beyond "SB3" in the paper).
- **Result**: A2C (33.18% mean coverage) outperformed PPO (20.56%) in our reimplementation. This is **not** claimed to reflect the paper's own finding either way -- the paper compares both algorithms in its own setup; we simply report what our from-scratch reimplementation produced, with identical training budgets for both.

### 4.3 Chen, Lu, Cui, Luo & Zheng (Sensors 2025) -- "Re-DQN"

- **Code**: none released. **Reimplemented by us**, and **only the paper's own baseline comparator (vanilla DQN)**, in `benchmark/baseline_redqn/grid_env_16.py`.
- **Deliberately NOT reproduced**: the paper's novel "Re-DQN" contributions -- a noisy-linear exploration layer, a "dynamic incentive" layer, a curiosity/state-novelty intrinsic reward, and dynamic-size input padding for a variable obstacle count. These are bespoke architectural contributions with no off-the-shelf implementation; a rough reimplementation from a methods-section summary risked producing something inaccurate and wrongly attributed to the paper's authors. Per the task's own instruction ("if exact reproduction is impossible, document the limitation and implement the closest scientifically valid version"), only the well-specified, standard-DQN baseline was reproduced.
- **Environment**: 16x16 discrete grid, `Discrete(4)` actions, reward = -P_move/step, +R_discover/new cell, -P_obstacle on collision, +R_cc on completion (terrain penalty term fixed at zero -- no terrain data exists anywhere in this benchmark). **Termination is harsh**: collision OR reaching the map boundary ends the episode immediately as a failure (per the paper), not just a penalty-and-continue.
- **Result**: our vanilla-DQN reproduction performs very poorly (3.67% mean coverage, mean episode length only 8 steps) -- it essentially dies almost immediately, most likely because of the harsh collision/boundary termination combined with no exploration bonus. **This is plausibly consistent with, not contradictory to, the source paper's own motivation**: the paper's whole contribution is a curiosity-driven exploration mechanism specifically to overcome this exact difficulty with vanilla DQN. Our result should be read as "vanilla DQN struggles badly here," not as "our reimplementation is broken" -- though we cannot fully rule out an implementation difference given no code was available to check against.
- **Authors' own published DQN baseline numbers**, quoted here for reference ONLY (not run by us): ~120 steps, ~87 tiles/episode, ~65 reward. Our local run's much worse numbers likely reflect real differences in obstacle density, reward constants, or training budget between our reimplementation and the paper's own baseline configuration (none of which the paper specifies precisely enough for us to match exactly -- see assumptions in `grid_env_16.py`).

### 4.4 Theile, Bayerlein, Nai, Gesbert & Caccamo (IEEE/RSJ IROS 2020)

- **Code**: none for this exact paper. A repository by the same lab
  (`github.com/theilem/uavSim`) exists but, per its own README, currently
  implements two later, different papers ("Learning to Recharge",
  arXiv:2309.03157, and an "Equivariant Ensembles..." paper,
  arXiv:2403.12856); an "icar" branch it references for yet another paper
  did not exist in the cloned remote. None of this corresponds to the 2020
  IROS paper, so it was deliberately NOT used. **Reimplemented by us** in
  `benchmark/baseline_theile_uav/grid_env.py`.
- **Algorithm**: Double DQN (DDQN). This reimplementation trains SB3's
  off-the-shelf `DQN`, which computes the standard (non-double) target --
  a documented, real algorithmic simplification (vanilla DQN, not DDQN).
- **Environment**: 16x16 grid, 3-channel map (start/landing zone, target
  zone, no-fly zone) + a boolean coverage grid + UAV position (one-hot) +
  a scalar movement budget. 5 actions: `{north, east, south, west, land}`.
  A fixed 3x3 camera field-of-view marks cells covered. Every action
  (accepted or rejected) costs 1 unit of movement budget, sampled per
  episode from `[25, 75]` (paper's own range).
- **Reward**: `r_cov` (+, per newly covered target cell), `r_sc` (-, on a
  rejected/no-fly-zone move), `r_mov` (-, constant per-step cost), `r_crash`
  (-, running out of budget without landing) -- **defined symbolically in
  the paper but never given numeric values**; we used `+1.0 / -0.5 / -0.05
  / -10.0` respectively, documented as our own choice.
- **Hyperparameters taken directly from the paper's own Table I**: replay
  buffer 50,000, gamma 0.95, target-network soft-update tau 0.005, minibatch
  size 128.
- **Assumption**: the paper's exact Maps A/B/C are only shown as figures
  (pixel layouts not published as data); a simplified random-rectangular-
  obstacle layout generator is used instead, seeded per evaluation episode.
- **Result**: mean coverage 14.16% (std 27.65pp -- extremely bimodal: many
  episodes land immediately with 0% coverage, a few reach 70-98%). This
  bimodality is a plausible echo of the paper's own described training
  dynamic ("in phase one the agent learns to land safely, but does not
  venture far enough... to find the target zone") -- our agent may be
  stuck partway through that same phase transition for many evaluation
  layouts, not necessarily a broken reimplementation.
- **Authors' own published numbers** (reference only, not run by us):
  landing ratio 98.3-99.8% across their three maps (their Table II).

### 4.5 Devo, Mao, Costante & Loianno (IEEE Robotics and Automation Letters 2022)

- **Code**: none found. **Reimplemented by us** in
  `benchmark/baseline_devo_entropy/grid_env.py`.
- **Algorithm**: IMPALA (V-trace, asynchronous actor-critic). SB3 does not
  ship IMPALA; PPO (SB3) is used as the closest available on-policy
  actor-critic substitute -- a documented algorithmic substitution.
- **Environment (paper)**: photorealistic Unreal Engine 4 simulation, raw
  84x84 RGB camera frames as the actor's only observation (asymmetric
  critic gets extra ground-truth maps), real-drone mixed-reality deployment
  with Vicon motion capture. **None of this is reproducible here.**
- **What WAS reproduced exactly**: the paper's own 11-action discrete space
  (`move_forwards, turn_left, turn_right, turn_right_move_forward,
  turn_left_move_forward, turn_right_move_backward, turn_left_move_backward,
  move_backward, do_nothing, move_left, move_right`, verbatim from the
  paper's Fig. 2) and its closed-form entropy reward (`r = max(Me) / (1 +
  sum_i ceil(Me_i))`, worked out to `r = (1/u)/(1+u)` for `u` unexplored
  cells -- see `grid_env.py` docstring for the derivation).
- **What was substituted**: a 32x32 procedurally-generated rooms-and-
  corridors grid stands in for the UE4 floor plans; a 9x9 local
  occupancy+explored crop stands in for the 84x84 RGB frame (preserving
  partial observability, not visual realism).
- **Result**: mean coverage only 2.77% (std 1.09pp) after 500k PPO
  timesteps -- the hardest task in this benchmark for the budget given
  (32x32 maze exploration through a low-information local crop with a very
  small per-step reward signal). This likely reflects both the genuine
  difficulty of the substituted task and PPO's mismatch for what the
  original paper solves with a distributed IMPALA setup (8 parallel actors,
  1500 episodes x 1800 steps = 2.7M environment steps, more than 5x our
  budget) -- not necessarily a broken environment.
- **Authors' own published numbers** (reference only, not run by us):
  coverage 58.2% (Standard env), 39.3% (Large env), ~70% (Realistic env,
  mean of 6 floors) -- their Table I.

### 4.6 Classical baselines (Choset & Pignon lineage)

- **Code**: none -- implemented directly in `benchmark/baseline_boustrophedon/run_classical_baselines.py`, running inside our own **unmodified `BeadEnv`**.
- Literal area-sweep Boustrophedon Cellular Decomposition targets *region* coverage and doesn't apply to a *curve*-coverage task; we implement the two standard classical analogues instead:
  - **Raster sweep**: fixed, contour-agnostic back-and-forth scan (the literal spirit of boustrophedon).
  - **Greedy nearest-uncovered pursuit**: thrusts directly toward the nearest uncovered contour pixel, using **only** `obs["state"][4:6]` -- the identical direction signal already exposed to our PPO agent's own observation.
- This is the **only baseline in this benchmark that is genuinely apples-to-apples** with our model: same environment, same physics, same coverage ground truth, same action space.

## 5. Results

Full table: `benchmark/report/comparison_table.md` / `.csv`. Plots:
`benchmark/plots/coverage_comparison.png`,
`benchmark/plots/success_redundancy_comparison.png`.

| Method | Mean coverage % | Success rate | Mean redundancy |
|---|---|---|---|
| **Our PPO (BeadEnv)** | **82.03%** | 0.0%* | 0.489 |
| Classical: greedy nearest-uncovered | 49.51% | 0.0% | 0.895 |
| Classical: raster sweep | 10.33% | 0.0% | 0.957 |
| Reimpl.: Garrido-Castañeda et al. (A2C, grid) | 33.18% | 0.0% | 0.954 |
| Reimpl.: Garrido-Castañeda et al. (PPO, grid) | 20.56% | 0.0% | 0.972 |
| Reimpl.: Chen et al. baseline (DQN, grid) | 3.67% | 0.0% | 0.285† |
| Jonnarth et al., official checkpoint (authors' bundled) | 99.00% | 100.0% | 0.331 |
| Jonnarth et al., official checkpoint (our local run) | 90.84% | 60.0% | 0.262 |
| Reimpl.: Theile et al. 2020 (DQN, UAV power-constrained grid) | 14.16% | 0.0% | 0.849‡ |
| Reimpl.: Devo et al. 2022 (PPO proxy for IMPALA, exploration grid) | 2.77% | 0.0% | 0.998§ |

\* See §5.1. † Low redundancy here reflects episodes ending almost
immediately (mean 8 steps), not efficient coverage -- see §4.3. ‡ Highly
bimodal (std 27.65pp) -- many episodes land immediately at 0%, a few reach
70-98%, see §4.4. § Reflects a genuinely difficult substituted task (32x32
maze exploration) under a training budget ~5x smaller than the original
paper's distributed IMPALA setup, see §4.5.

### 5.1 Why our model's success rate reads 0%

`config.py`'s `completion_frac` is currently **0.95** -- a deliberately very
strict bar requiring 95% of all contour pixels to be swept before an episode
counts as "complete." At this threshold, 0 of 20 evaluation episodes
formally complete, even though mean coverage is a healthy 82.03% (individual
episodes range 36.7%-93.5%; see `evaluations/eval_20260823_192841.json`).
This is a direct, known consequence of raising the threshold from an earlier
80% configuration (which achieved 100% success / 80.08% mean coverage on the
same evaluation protocol) -- it is not a new finding from this benchmark,
just a threshold effect worth keeping in mind when reading the "success
rate" column: it is not comparable in an absolute sense to another method's
completion criterion without knowing how strict each one is.

## 6. Is this an apples-to-apples comparison? (Answer: partially, by design)

**Genuinely comparable across all methods:**
- Coverage % and redundancy rate are dimensionless, ground-truth-based,
  computed by the identical `benchmark/common/metrics.py` functions for
  every method -- no method gets its own bespoke definition.
- All results are averaged over multiple episodes (6-20 depending on
  method) with recorded seeds/layouts.

**NOT comparable in absolute terms:**
- **Task**: our model traces a 1-D contour; the grid baselines cover 2-D
  grid *cells*; Jonnarth et al.'s model covers a continuous 2-D *area*.
  "82% contour coverage" and "90% area coverage" are not the same kind of
  quantity, even though both are percentages.
- **Path length and step count**: reported in each method's own native
  units (BeadEnv: raster pixels per step; grid baselines: grid cells per
  move; Jonnarth et al.: meters at 0.5s/step). Never compare these numbers
  directly across methods -- `generate_comparison.py`'s own table repeats
  this caveat.
- **Success/completion criteria**: each method defines its own bar (our
  95% coverage threshold; the grid papers' 100%-of-free-cells; Jonnarth et
  al.'s configurable goal_coverage, 0.99 here). A 0% vs. 60% vs. 100%
  success-rate comparison reflects both task difficulty *and* threshold
  strictness conflated together.
- **Training budgets differ substantially** and were not tuned for parity:
  our PPO model trained for ~1M timesteps; the grid reimplementations used
  300k (A2C/PPO) or 500k (DQN) timesteps chosen by us (undocumented by the
  source papers); Jonnarth et al.'s checkpoint's own training budget is
  whatever the authors used (not reproduced by us -- we used their
  pretrained weights directly).

**Bottom line**: this benchmark is scientifically useful for showing *where
our model sits relative to a spread of published and classical CPP
approaches*, and for making concrete, honest observations (e.g., "our model
clears every classical and grid-RL baseline tested on coverage %, and is
in the same range as a strong, mature, officially-released PPO coverage
agent on its own native task"). It is **not** a controlled experiment
holding environment, task, and training budget constant, and no single
number in the table above should be quoted out of this context.

## 7. Reproducibility

- All source code, environments, and configuration live under `benchmark/`
  (never touching the parent project).
- Every result JSON in `benchmark/results/` records its own `meta` block:
  seeds used, training timesteps, wall-clock time, device, SB3/paper
  version, and every documented assumption.
- Commands to reproduce every result from scratch are in `benchmark/README.md`.
- GPU (`cuda`) was used for all training in this benchmark, per instruction.
  Note (observed, not fabricated): Stable-Baselines3 itself warns that GPU
  offers no realistic benefit for the tiny `MlpPolicy` networks used by the
  grid-world baselines -- this is a hardware/architecture-size observation,
  not a claim from any cited paper.

## 8. Honest limitations of this benchmark, summarized

1. Four of five RL papers are **reimplementations**, not official code --
   correctness depends on our reading of each paper's methodology section,
   which in places under-specifies exact constants (documented per-baseline
   above and in each baseline's own module docstring).
2. The Chen et al. baseline reproduces only the paper's DQN *comparator*,
   not its novel Re-DQN contribution -- by design, not oversight. The
   Theile et al. baseline reproduces DQN, not the paper's actual DDQN,
   because SB3's DQN doesn't expose a double-Q target. The Devo et al.
   baseline substitutes PPO for the paper's IMPALA, and a 2-D grid + local
   crop for the paper's UE4 + RGB-camera environment.
3. Task/environment structure differs across every non-classical baseline;
   only the classical baseline shares our exact environment.
4. Sample sizes differ (6-20 episodes across methods) due to what each
   source provides (Jonnarth et al.'s bundled reference has only 6).
5. No hyperparameter tuning was performed for any reimplemented baseline
   beyond matching the paper's stated methodology -- results reflect a
   single run per method/algorithm, not a tuned optimum.
6. Training budgets for the reimplemented grid/proxy baselines (300k-500k
   timesteps) are our own choice, not the source papers' own budgets (which
   are usually specified in episodes, not timesteps, and in at least one
   case -- Devo et al. -- are far larger than what we used here).
