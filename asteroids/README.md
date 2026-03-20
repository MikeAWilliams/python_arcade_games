# Asteroids

An Asteroids game implementation in Python with pluggable AI input system. The project supports both interactive gameplay and headless multi-process AI benchmarking.

## Project Structure

```
asteroids/
├── asteroids/              # Main package
│   ├── core/              # Core game logic and base input
│   │   ├── game.py        # Game engine, entities, Action, InputMethod
│   │   ├── game_runner.py # Headless game execution
│   │   └── keyboard.py    # Keyboard input implementation
│   └── ai/                # AI-based input implementations
│       ├── heuristic.py   # Rule-based AI with genetic optimization
│       ├── raw_geometry_nn.py  # Neural network AI (raw geometry input, 142 features)
│       └── polar_nn.py    # Neural network AI (polar-coordinate features, 39 features)
│
├── training/              # Training scripts
│   ├── policy_gradient.py # Policy gradient (REINFORCE) training
│   ├── genetic.py         # Genetic algorithm optimization
│   └── cross_entropy.py   # Supervised learning (cross-entropy)
│
├── tools/                 # Utility scripts
│   ├── compact_recordings.py    # Consolidate game recordings
│   ├── analyze_state_data.py    # Inspect training data column stats
│   ├── convert_training_data.py # Convert angle to bearing format
│   ├── generate_random_model.py # Create a fresh random model file
│   ├── measure_entropy.py         # Measure policy entropy and action distribution
│   ├── visualize_state.py         # Visualize raw geometry state recordings
│   ├── visualize_reward_shaping.py # Visualize death penalty reward shaping
│   └── visualize_state_polar.py   # Visualize polar state (converted from raw)
│
├── data/                  # Training data (not in git)
├── nn_checkpoints/        # Training checkpoints and logs (not in git)
├── nn_weights/            # Trained model weights (not in git)
│
├── main_arcade.py         # Interactive game (keyboard/AI)
├── main_headless.py       # Headless benchmarking
```

### Data Directories (not in git)

- **`data/`** - Recorded gameplay stored as `.npz` files. Created by `main_headless.py --record`. Contains state vectors, actions, game IDs, and tick numbers used for supervised training.
- **`nn_checkpoints/`** - Intermediate checkpoints and log files saved during cross-entropy training. Checkpoints are saved at regular intervals so training can be inspected or resumed.
- **`nn_weights/`** - Final trained model weights (`.pth` files) ready for use with `main_arcade.py --air`/`--aip` or `main_headless.py --ai-type raw`/`polar`.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Interactive Mode (Play the Game)

```bash
# Play with keyboard controls
python main_arcade.py

# Watch heuristic AI play
python main_arcade.py --aih

# Watch raw geometry neural network AI play (requires trained model)
python main_arcade.py --air
python main_arcade.py --air custom_model.pth

# Watch polar neural network AI play (requires trained model)
python main_arcade.py --aip
python main_arcade.py --aip custom_polar.pth
```

**Keyboard Controls:**
- Arrow Keys: Turn left/right, thrust forward/backward
- Space: Shoot

### Headless Benchmarking

Run multiple games in parallel to evaluate AI performance:

```bash
# Run 10 games with heuristic AI
python main_headless.py -n 10

# Run 100 games on 8 CPU cores
python main_headless.py -n 100 -t 8

# Benchmark raw geometry neural network AI
python main_headless.py --ai-type raw --model-path model.pth -n 50

# Benchmark polar neural network AI
python main_headless.py --ai-type polar -n 50

# Record gameplay data (with logging)
python main_headless.py -n 10 --record heuristic_data
```

#### Recording Outputs

When using the `--record <basename>` option, headless benchmarking creates:

- **Recording Files**: `data/<basename>_<game_id>.npz`
  - Individual game recordings with state and action data
  - Used for training data collection

- **Log File**: `data/<basename>.log`
  - Complete console output from the recording session
  - Includes run configuration, progress updates, and final statistics
  - Overwritten on each new recording run with the same basename (no timestamp)
  - Example: `--record heuristic_data` creates `data/heuristic_data.log`

### Training

**Train Neural Network (Policy Gradient):**
```bash
python training/policy_gradient.py --batch-size 32 --workers 8
```

**Train with Supervised Learning (Cross-Entropy):**
```bash
python training/cross_entropy.py --base-name heuristic_data --batch-size 64 --epochs 200
```

**Optimize Heuristic AI (Genetic Algorithm):**
```bash
python training/genetic.py --population 50 --generations 20
```

#### Training Outputs

**Neural Network Training** (`training/policy_gradient.py`):

All artifacts saved to `nn_checkpoints/`:

- **Log Files**: `training_YYYYMMDD_HHMMSS.log`
  - Complete console output from training run
  - Includes progress, scores, timing information

- **Checkpoints**: `checkpoint_YYYYMMDD_HHMMSS_*.pth`
  - `epoch_N.pth`: Periodic checkpoints (every 6000 epochs)
  - `final.pth`: Final model after all epochs

- **Checkpoint Contents**:
  - Model weights (`model_state_dict`)
  - Optimizer state (`optimizer_state_dict`)
  - Training metadata (epoch, max_score, loss)

**Genetic Algorithm Training** (`training/genetic.py`):

Log file saved to root directory:

- **Log File**: `genetic.log`
  - Complete console output from optimization run
  - Includes generation progress and best parameters
  - Overwritten on each new run

**Cross-Entropy Training** (`training/cross_entropy.py`):

All artifacts saved to `nn_checkpoints/`:

- **Log Files**: `<basename>_YYYYMMDD_HHMMSS.log`
  - Complete console output from training run
  - Includes data loading, training progress, errors

- **Checkpoints**: `<basename>_YYYYMMDD_HHMMSS_*.pth`
  - Saved during training (user implements checkpoint logic)
  - Final model saved as `<basename>_YYYYMMDD_HHMMSS_final.pth`

**Note**: The `TRAINING_RUN_NAME` constant at the top of `cross_entropy.py` controls all output file names. Set it before each new training regime.

#### Cross-Entropy Training Runs

| Run name | Log file | Result | Notes |
|----------|----------|--------|-------|
| `training_data20k_combinded` | `training_data20k_combinded_cross_entropy.log` | — | Old run. State used a scalar normalized angle (`angle / 2π`) at column 4. 141 input features. |
| `training_data20k_converted` | `training_data20k_converted_cross_entropy.log` | Avg score ~9–10 | Bearing format (cos/sin at cols 4–5, 142 inputs). No class weights. Model learned to spin but rarely shot — action probabilities mirrored the training distribution (~93% turns) so shooting was almost never the argmax. |
| `bearing_weighted` | `bearing_weighted_cross_entropy.log` | Avg score ~2–7 | Same bearing format with full inverse-frequency class weights (TURN: 0.05×, ACCEL: 3.07×, SHOOT: 0.57×). **Failed.** Down-weighting turns to 0.05 starved the model of survival signal. It learned to accelerate and decelerate without regard to asteroid position — the upweighted minority actions were reinforced, but the model never associated them with relevant game states. |

#### Policy Gradient Training Runs

| Run name | Model | Log file | Result | Notes |
|----------|-------|----------|--------|-------|
| `polar_pg` | polar | `polar_pg_policy_gradient.log` | Avg ~120, max ~140 | First successful PG run. Learned to aim and shoot. Converged to stationary turret strategy (left/right turn + shoot, no movement). Explosive score jump around iter 29k. |
| `polar_pg_entropy` | polar | `polar_pg_entropy_policy_gradient.log` | Avg ~80–120 | Added entropy bonus (0.01) to break local optimum. Scores dropped initially as policy diversified. Inconclusive — stopped to focus on polar2. |
| `polar2_pg` | polar2 | `polar2_pg_policy_gradient.log` | Avg ~40, max ~70 | Polar2 model (edge distance, TTI, lateral velocity, player velocity direction) from scratch. Learned slower than polar v1. Policy collapsed to two actions (left-turn 52%, shoot 47%) — entropy at 9% of max. Plateaued at ~40 avg. |
| `polar2_pg_entropy` | polar2 | `polar2_pg_entropy_policy_gradient.log` | Avg ~90–110, max ~128 | Resumed from `polar2_pg_best.pth` with entropy bonus (0.01). Broke out of local optimum within ~1500 iterations. All 6 actions now used. Entropy rose to 77% of max. |
| `polar2_pg_exploit` | polar2 | `polar2_pg_exploit_policy_gradient.log` | Best 136, avg ~50–60 | Resumed from `polar2_pg_entropy_best.pth` with entropy=0. Hit best 136.41 at iter 1137, then avg scores collapsed to 50–60 range for remaining 35k iters. Policy narrowed to aggressive charge-and-shoot strategy — high variance, no further improvement. Stopped at iter 36k. |
| `polar2_pg_ent005` | polar2 | `polar2_pg_ent005_policy_gradient.log` | **Best 159, avg ~100–120** | **Current best run.** Resumed from `polar2_pg_exploit_best.pth` (136.41) with entropy=0.005. Steady improvement over ~21k iters: best 136→142→147→159. Avg scores healthy at 100–120 without the collapse seen in exploit run. Plateaued after iter ~15k. |
| `polar2_pg_ent005_dp` | polar2 | `polar2_pg_ent005_dp_policy_gradient.log` | No improvement | Resumed from `polar2_pg_ent005_best.pth` (159.05) with entropy=0.005 and death penalty (-0.5 over 60 frames). Ran 15k iters over ~5 hours with zero new bests. Avg scores same 90–125 range. Penalty too strong: cumulative -15 over 60 frames overwhelmed the +1 kill rewards, drowning out the scoring signal. |
| `polar2_pg_ent005_dp01` | polar2 | `polar2_pg_ent005_dp01_policy_gradient.log` | — | Resumed from `polar2_pg_ent005_best.pth` (159.05) with entropy=0.005 and reduced death penalty (-0.1 over 60 frames). Cumulative penalty ~-3, more proportionate to +1 kill rewards. |

### Utilities

**Consolidate Recording Files:**
```bash
python tools/compact_recordings.py
```

**Analyze Training Data Columns:**
```bash
python tools/analyze_state_data.py
python tools/analyze_state_data.py --base-name training_data20k_v2
python tools/analyze_state_data.py --sample-size 100000 --file-index 3
```
Shows per-column min/max/mean/std for state vectors in training data files.

**Convert Training Data (Angle to Bearing):**
```bash
python tools/convert_training_data.py --input-base training_data20k_combinded --output-base training_data20k_v2
```
Converts old-format training data (scalar angle at column 4) to new bearing format (cos/sin at columns 4-5). Required after the angle-to-bearing encoding change.

**Generate Random Model:**
```bash
python tools/generate_random_model.py                        # raw geometry (nn_weights/nn_model.pth)
python tools/generate_random_model.py --model polar          # polar (nn_weights/nn_polar.pth)
python tools/generate_random_model.py --model polar --output nn_weights/my_model.pth
```
Creates a randomly-initialized model weights file. Useful for bootstrapping when no trained model exists.

**Measure Policy Entropy:**
```bash
python tools/measure_entropy.py                                              # defaults to nn_checkpoints/polar_pg_best.pth
python tools/measure_entropy.py --checkpoint nn_weights/polar_pg_best.pth    # specify checkpoint
python tools/measure_entropy.py --max-frames 10000                           # run longer game
```
Reports per-frame entropy statistics and average action probabilities to assess how deterministic the policy has become.

**Visualize Reward Shaping:**
```bash
python tools/visualize_reward_shaping.py
```
Shows the effect of death penalty reward shaping on discounted returns — compares raw rewards, the penalty ramp, and normalized advantages with and without the penalty.

**Visualize State Recordings:**
```bash
python tools/visualize_state.py                  # raw geometry state
python tools/visualize_state_polar.py            # polar state (converted from raw recordings)
```

## AI Implementations

### Heuristic AI
Rule-based AI using:
- **Evasive maneuvers** - Weighted threat vectors from nearby asteroids
- **Speed control** - Target velocity management
- **Predictive targeting** - Intercept calculations for shooting

### Neural Network AI — RawGeometry
Feeds raw game geometry directly to the network (142 inputs):
- State: Player x/y, velocity, bearing (cos/sin), cooldown + 27 asteroid slots (x/y, velocity, active flag)
- Architecture: 142 → 128 (ReLU) → 6
- Actions: Turn left/right, accelerate/decelerate, shoot, no action

### Neural Network AI — Polar
Pre-computes player-relative features (39 inputs):
- Global: player speed, shot cooldown, asteroid count
- Per asteroid (top 9 by distance): distance, relative angle from bearing, closing speed, size category
- Architecture: 39 → 128 (ReLU) → 64 (ReLU) → 6
- Includes `convert_raw_geometry_state()` to reuse existing training data

## Related Work

Survey of known prior approaches to Asteroids AI. See `research_notes.md` for detailed notes and links.

| Source | Model size | Input to model | Training |
|--------|-----------|----------------|----------|
| Cuccu et al. — "Six Neurons" (2019) | 6–18 neurons (policy net only) | IDVQ-compressed encoding (handful of numbers) | ENES |
| pjflanagan — Machine Learning Asteroids | 8→6→4 | 8 raycasting distances, one per compass direction | GA |
| Immodal — NEAT Asteroids | Evolved by NEAT (starts minimal) | Raycasting distances in multiple directions | NEAT |
| Hausknecht et al. (2014) | Evolved | Object features (likely raw pos/vel from RAM), raw pixels, or noise — compared all three | NEAT, HyperNEAT, CMA-ES, NE |
| jrtechs blog | 2 conv layers + 256-unit FC | Raw pixels | DQN (failed — implementation bugs) |
| datadaveshin — AIsteroids | Tabular (no network) | Discrete attributes (is asteroid near, collision status) | Q-learning |
| hdparks — AsteroidsDeepReinforcement | Unknown | Unknown | PPO |
| Machneva — Medium/data-surge | Unknown (MLP policy) | Gymnasium state, likely RAM-based given MLP choice | PPO (SB3) |
| lgoodridge — Asteroids-AI | Unknown | Unknown | GA |
| Delfosse et al. — OCAtari (2023) | Various RL agents | Structured object state from RAM: position, size, colour per object | Various RL |

**Acronyms:**
BC — Behavioural Cloning |
CMA-ES — Covariance Matrix Adaptation Evolution Strategy |
DQN — Deep Q-Network |
ENES — Exponential Natural Evolution Strategies |
FC — Fully Connected layer |
GA — Genetic Algorithm |
IDVQ — Increasing Dictionary Vector Quantization |
MLP — Multi-Layer Perceptron |
NEAT — NeuroEvolution of Augmenting Topologies |
HyperNEAT — Hypercube-based NEAT |
NE — NeuroEvolution |
PPO — Proximal Policy Optimisation |
RAM — Random Access Memory (Atari hardware state) |
RL — Reinforcement Learning |
SB3 — Stable Baselines 3
