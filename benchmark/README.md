# Benchmark: comparing our PPO bead-tracer against published CPP baselines

This directory is a **separate, self-contained benchmark framework**. Nothing
in it modifies the parent project (`env.py`, `config.py`, `train.py`,
`evaluate.py`, `inference.py`, the trained model, or any existing results) --
it only reads the trained model and calls the environment's existing public
interface, exactly like `evaluate.py` does.

## Why these baselines

Full selection rationale (including papers considered and rejected) is in
`report/report.md`. In short, 3 published RL papers plus 1 classical
algorithm were chosen based on relevance and reproducibility, not on which
had the flashiest reported numbers:

| # | Method | Source | Reproduction |
|---|---|---|---|
| 1 | PPO, map-based CNN observation, coverage task | Jonnarth, Zhao & Felsberg, ICML 2024 / Jonnarth, Johansson, Zhao & Felsberg, IEEE Access 2025 ("rl-cpp") | **Official code + official pretrained weights**, run locally by us |
| 2 | A2C and PPO, discrete grid, via Stable-Baselines3 | Garrido-Castaneda, Vasquez & Antonio-Cruz, Sensors 2025 | **Reimplemented by us** (no official code released) |
| 3 | DQN (paper's own baseline, not their novel Re-DQN) | Chen, Lu, Cui, Luo & Zheng, Sensors 2025 | **Reimplemented by us** (no official code released; only the paper's baseline comparator is reproduced, not its novel contributions -- see `baseline_redqn/grid_env_16.py`) |
| 4 | DQN (paper's own DDQN, approximated), UAV coverage under power constraint | Theile, Bayerlein, Nai, Gesbert & Caccamo, IEEE/RSJ IROS 2020 | **Reimplemented by us** (no code for this exact paper; a same-lab repo exists but implements later, different papers -- see `baseline_theile_uav/grid_env.py`) |
| 5 | PPO (proxy for the paper's IMPALA), single-image drone exploration | Devo, Mao, Costante & Loianno, IEEE RA-L 2022 | **Reimplemented by us** (paper's own UE4+drone+Vicon setup not reproducible; exact 11-action space and entropy reward kept -- see `baseline_devo_entropy/grid_env.py`) |
| 6 | Boustrophedon-derived classical baselines (raster sweep + greedy pursuit) | Choset & Pignon, 1997/2000 (classical robotics, not a DRL paper) | **Implemented directly inside our own unmodified BeadEnv** -- the only baseline that is genuinely apples-to-apples with our model |

## Paper copies

Each baseline folder contains the actual paper it reproduces (`paper_*.pdf`,
or `paper_*.html` for the two MDPI/Sensors papers -- see note below):

| Folder | File(s) |
|---|---|
| `baseline_rlcpp_external/` | `paper_jonnarth_icml2024.pdf`, `paper_jonnarth_ieeeaccess2025.pdf` (both arXiv versions of the ICML 2024 paper and its IEEE Access 2025 journal extension) |
| `baseline_sensors_actorcritic/` | `paper_garrido-castaneda_sensors2025.html` (full PMC article page) |
| `baseline_redqn/` | `paper_chen_sensors2025.html` (full PMC article page) |
| `baseline_theile_uav/` | `paper_theile_iros2020.pdf` (arXiv version) |
| `baseline_devo_entropy/` | `paper_devo_ral2022.pdf` (accepted-manuscript version) |
| `baseline_boustrophedon/` | `paper_choset_pignon_2000.pdf` (the Autonomous Robots 2000 journal version, freely hosted by CMU) |

**Why two papers are `.html`, not `.pdf`**: both MDPI/Sensors papers'
publisher-side PDF endpoints are protected against automated downloads --
MDPI's own site returns an outright 403 to non-browser requests, and its
PMC mirror serves a JavaScript proof-of-work challenge page instead of the
file. Neither is solvable by a plain HTTP request. The `.html` files are
the complete, genuine open-access article pages (verified to contain the
full paper text, not an error/challenge page) fetched from PMC, which does
not protect its plain article view the same way. If you need an actual PDF
for these two, open the DOI in a real browser: 10.3390/s25051592 and
10.3390/s25020416.

## Directory structure

```
benchmark/
├── common/
│   └── metrics.py                 # ONE shared metrics implementation used
│                                   # by every method below -- no baseline
│                                   # gets its own bespoke metric definition
├── evaluate_ours.py                # Runs our own trained PPO model through
│                                   # this benchmark's metrics (adds
│                                   # path_length/redundancy on top of
│                                   # evaluate.py's existing numbers)
├── baseline_boustrophedon/
│   └── run_classical_baselines.py  # Raster-sweep + greedy-pursuit, inside
│                                   # our own unmodified BeadEnv
├── baseline_sensors_actorcritic/
│   ├── grid_env.py                 # Reimplementation of the paper's 9x9
│                                   # discrete grid environment
│   └── train_and_evaluate.py       # Trains + evaluates A2C and PPO via SB3
├── baseline_redqn/
│   ├── grid_env_16.py              # Reimplementation of the paper's 16x16
│                                   # grid + BASELINE reward (not Re-DQN)
│   └── train_and_evaluate_dqn.py   # Trains + evaluates vanilla SB3 DQN
├── baseline_theile_uav/
│   ├── grid_env.py                 # Reimplementation of Theile et al. 2020's
│                                   # power-constrained UAV coverage grid
│   └── train_and_evaluate.py       # Trains + evaluates vanilla SB3 DQN
├── baseline_devo_entropy/
│   ├── grid_env.py                 # Reimplementation of Devo et al. 2022's
│                                   # 11-action space + entropy reward, in a
│                                   # 2-D grid proxy (not UE4/RGB)
│   └── train_and_evaluate.py       # Trains + evaluates SB3 PPO
├── jonnarth_map_on_beadenv/
│   ├── jonnarth_eval_mowing_9.png   # a Jonnarth et al. benchmark map image
│                                   # (from the official rl-cpp repo), used
│                                   # as a TARGET IMAGE for our own BeadEnv
│                                   # -- NOT running our model inside
│                                   # MowerEnv (impossible, see report.md §5.2)
│   ├── evaluate.py                 # evaluates our model on this map image
│   └── simulate.py                 # animated replay, same as project's own
├── theile2024_map_on_beadenv/
│   ├── theile2024_tum50.png         # a real map asset from Theile et al.'s
│                                   # IEEE IROS 2024 official repo
│                                   # (theilem/uavSim), used as a TARGET
│                                   # IMAGE for our own BeadEnv -- NOT
│                                   # running our model inside their
│                                   # CPPGym (impossible, discrete vs
│                                   # continuous action space -- see
│                                   # report.md §5.3)
│   ├── evaluate.py                 # evaluates our model on this map image
│   └── simulate.py                 # animated replay
├── f1tenth_racetrack_on_beadenv/
│   ├── hockenheim.png / montreal.png / yasmarina.png
│                                   # F1TENTH racetrack outlines (from
│                                   # github.com/f1tenth/f1tenth_racetracks,
│                                   # the exact dataset used by Elgouhary &
│                                   # El-Wakeel, arXiv:2602.18386) -- single
│                                   # continuous closed loops, used as
│                                   # TARGET IMAGES for our own BeadEnv, NOT
│                                   # running our model inside F1TENTH Gym
│                                   # (impossible -- see report.md §5.4)
│   ├── evaluate.py                 # evaluates our model on all 3 tracks
│   └── simulate.py                 # animated replay (--track hockenheim/
│                                   # montreal/yasmarina)
├── baseline_rlcpp_external/
│   ├── parse_official_eval.py      # parses mowing_tv1 eval CSVs into our schema
│   ├── render_exploration_episode.py  # renders ONE episode of the official
│                                   # "exploration" checkpoint (SAC, not
│                                   # PPO) to a GIF via the environment's
│                                   # own rgb_array renderer
│   ├── render_mowing_episode.py    # same idea, for the official "mowing"
│                                   # checkpoints (also SAC, despite the
│                                   # paper describing PPO -- see the
│                                   # script's docstring), forced onto ONE
│                                   # specific named eval map via
│                                   # env.eval_maps = [that map] before reset()
│   └── official_code/              # Cloned official repo (arvijj/rl-cpp),
│                                   # run in its own isolated conda env
│                                   # (rlcpp_baseline, Python 3.9) because it
│                                   # requires gym==0.21.0 / SB3==1.6.2 --
│                                   # incompatible with this project's own
│                                   # Gymnasium/SB3 versions
├── results/                        # <method>.json + <method>.csv, one pair
│                                   # per method, written by save_results()
├── plots/                          # comparison_table's rendered charts
├── report/
│   ├── comparison_table.md/.csv    # generated by generate_comparison.py
│   └── report.md                   # full written report (methodology,
│                                   # results, limitations, apples-to-apples
│                                   # assessment)
└── generate_comparison.py          # reads results/*.json -> table + plots
```

## Running everything

```bash
# From the project root, using the project's OWN venv (.venv) for the
# baselines that share its dependencies:

# Our own model, through this benchmark's metrics
.venv\Scripts\python.exe benchmark\evaluate_ours.py

# Classical baselines (uses BeadEnv directly -- same env as our model)
.venv\Scripts\python.exe benchmark\baseline_boustrophedon\run_classical_baselines.py

# Sensors et al. 2025 reimplementation (A2C + PPO on a reimplemented grid)
.venv\Scripts\python.exe benchmark\baseline_sensors_actorcritic\train_and_evaluate.py --device cuda

# Chen et al. 2025 baseline reimplementation (vanilla DQN on a reimplemented grid)
.venv\Scripts\python.exe benchmark\baseline_redqn\train_and_evaluate_dqn.py --device cuda

# Theile et al. 2020 IROS reimplementation (DQN, UAV power-constrained grid)
.venv\Scripts\python.exe benchmark\baseline_theile_uav\train_and_evaluate.py --device cuda

# Devo et al. 2022 RA-L reimplementation (PPO, entropy-reward exploration grid)
.venv\Scripts\python.exe benchmark\baseline_devo_entropy\train_and_evaluate.py --device cuda

# rl-cpp official code -- needs its OWN isolated environment (see below)
conda activate rlcpp_baseline
cd benchmark\baseline_rlcpp_external\official_code
python eval.py --load weights\weights\mowing_tv1 --no-render --verbose --metrics_dir metrics_mowing_tv1 --steps 5000
cd ..\..\..
python benchmark\baseline_rlcpp_external\parse_official_eval.py

# Render one episode of the OFFICIAL "exploration" checkpoint (note: this
# checkpoint uses SAC, not PPO -- see the script's own docstring) to a GIF,
# using their own environment's built-in renderer:
cd benchmark\baseline_rlcpp_external\official_code
python ..\render_exploration_episode.py --steps 3000 --out ..\..\results\ppo_jonnarth_exploration_official.gif

# Same, for the official "mowing" checkpoint, forced onto one specific eval map:
python ..\render_mowing_episode.py --map maps\eval_mowing_7.png --steps 3000 --out ..\..\results\ppo_jonnarth_mowing_eval_mowing_7.gif
python ..\render_mowing_episode.py --map maps\eval_mowing_10.png --steps 3000 --out ..\..\results\ppo_jonnarth_mowing_eval_mowing_10.gif

# Once all/any of the above have run, generate the comparison table + plots:
.venv\Scripts\python.exe benchmark\generate_comparison.py
```

### Setting up the isolated `rlcpp_baseline` environment

The official `rl-cpp` code pins `gym==0.21.0` and `stable-baselines3==1.6.2`,
which conflict with this project's own `gymnasium`/`stable-baselines3`
versions and do not install cleanly under Python 3.13 (this project's
interpreter). It was installed in a separate conda environment:

```bash
conda create -n rlcpp_baseline python=3.9 -y
conda activate rlcpp_baseline
pip install "pip<24.1" "setuptools==59.5.0"
pip install --no-build-isolation gym==0.21.0
pip install -r benchmark/baseline_rlcpp_external/official_code/requirements.txt
pip install --force-reinstall torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```

(The two extra pins beyond the official README's own instructions --
`pip<24.1` and `setuptools==59.5.0` -- were needed to work around a known
packaging-metadata incompatibility between `gym==0.21.0`'s `setup.py` and
modern `pip`/`setuptools`; without them, `pip install gym==0.21.0` fails
outright on a current toolchain. This is a real, verifiable installation
issue, not an assumption.)

## Reproducibility

- All seeds used are recorded in each method's `results/<method>.json`
  under `meta`.
- Every training run's wall-clock time and device are recorded in the same
  `meta` block.
- No result in this benchmark is hand-edited; `generate_comparison.py`
  reads only from the saved JSON files.
