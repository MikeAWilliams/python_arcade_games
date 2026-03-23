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
│   ├── benchmark.sh               # Benchmark a polar2 model via headless runs
│   ├── measure_entropy.py         # Measure policy entropy and action distribution
│   ├── promote_checkpoint.sh      # Copy most recent checkpoint to nn_weights/
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

# Crisis mode (small asteroid dodge scenarios)
python main_arcade.py --crisis
python main_arcade.py --crisis --aih
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

# Mixed crisis training (10% crisis dodge scenarios + 90% normal games)
python training/policy_gradient.py --crisis-mix 0.1 --model-type polar2 --checkpoint nn_weights/model.pth --run-name my_crisis_run
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
| `polar2_pg_ent005_dp01` | polar2 | `polar2_pg_ent005_dp01_policy_gradient.log` | Best 163, avg ~118 (100-game) | Resumed from `polar2_pg_ent005_best.pth` (159.05) with entropy=0.005 and reduced death penalty (-0.1 over 60 frames). Two new batch bests (160, 163) but 100-game benchmark ~118, similar to ent005. Some subtle dodging observed visually but not enough to move the needle. |
| `polar2_pg_wave5` | polar2 | `polar2_pg_wave5_policy_gradient.log` | Best batch 83, **avg ~131 (1000-game)** | Curriculum training: games start at wave 5 (speed multiplier 1.46x). Resumed from `polar2_pg_ent005_best.pth` with entropy=0.005, death penalty (-0.1 over 60 frames). Two runs totaling ~52.5k iters (~7.5 hours): first 37.5k iters, then continued from checkpoint 36k for another 15k. 1000-game wave-1 benchmarks: avg 128 at 12k, **avg 131 at 18k** (peak), avg 128 at 24k (regressed). Entropy causes noisy action selection — rapid accel/decel and turn flickering observed visually. |
| `polar2_pg_wave5_exploit` | polar2 | `polar2_pg_wave5_exploit_policy_gradient.log` | Best batch 96, **avg ~130 (1000-game)** | Resumed from `polar2_pg_wave5_checkpoint_18000.pth` (131 avg) with entropy=0, death penalty (-0.1 over 60 frames), wave 5 curriculum. ~39.5k iters (~6.3 hours). No score improvement but eliminated noisy action flickering — no more useless accel/decel and turn toggling. Cleaner policy, same results. |
| `polar2_crisis` | polar2 | `polar2_crisis_policy_gradient.log` | **avg ~146.4 (1000-game)** | **100% crisis training — catastrophic forgetting.** *(affected by reward noise bug, see below)* Small-asteroid dodge scenarios (1–5 asteroids, timed to arrive before player can turn to face them). Resumed from `polar2_pg_wave5_exploit_best.pth` (130 avg). Best eval 151.21 at iter 1300 then steady collapse: 102→53→17→5 by iter 5500. Model became terrified of asteroids — learned to run instead of shoot. Crisis-only training overwrites normal gameplay policy. Best checkpoint (iter 1300) benchmarked at **146.4 avg over 1000 games** (all-time best). Led to crisis-mix approach. |
| `polar2_crisis_mix` | polar2 | `polar2_crisis_mix_policy_gradient.log` | No improvement | Mixed crisis training attempts. Concatenated crisis + normal game frames into single training batch. Crisis games (~100 frames) drowned out by normal games (~5000 frames) — crisis signal was <1% of gradient. 10% mix collapsed from 136→101 in 500 iters. 50% mix showed no improvement. Fixed by splitting into separate gradient updates per game type. |
| `polar2_crisis_mix97` | polar2 | `polar2_crisis_mix97_policy_gradient.log` | Best eval 155.21 at iter 1300, **no sustained improvement** | *(affected by reward noise bug, see below)* 97% crisis + 3% normal (31 crisis + 1 normal per batch). Separate gradient updates — normal and crisis each get their own training step so crisis signal isn't diluted by frame count disparity. Resumed from `polar2_pg_wave5_exploit_best.pth` (130 avg). Eval every 100 epochs on 100 normal games. ~23k iters (~1.5 hours). Wildly unstable — train_avg swung 11–290, eval_avg oscillated 50–140. Best eval 155.21 hit at iter 1300 (same timing as pure crisis run) then never improved. Separate gradients didn't prevent instability with 97% crisis mix. The high train_avg peaks (290+) were likely lucky easy spawns rather than learned dodging — a second run with best_training checkpoint saved (train_avg 329 at iter 647) showed the model just occasionally shooting single asteroids that spawned close enough to hit, not actually clearing waves consistently. The 18k checkpoint (only timed save) averaged 86 over 1000 games — worse than the starting point. Visual inspection showed it trying to shoot during crisis instead of dodging. |
| `polar2_crisis_mix97_v2` | polar2 | `polar2_crisis_mix97_v2_policy_gradient.log` | **Collapse — eval 146→78** | 97% crisis + 3% normal (31 crisis + 1 normal per batch) with reward noise fix. Resumed from `polar2_pg_wave5_exploit_best.pth` (130 avg). Best eval 146.67 at iter 300, then collapsed: 120→78 by iter 1000. Same catastrophic forgetting pattern as all other crisis-heavy runs — reward noise fix didn't help because the problem is overwhelming the normal-game policy with crisis gradients, not noise from failed crisis games. 1 normal game per batch is not enough to anchor the policy. |
| `polar2_crisis_mix100` | polar2 | `polar2_crisis_mix100_policy_gradient.log` | **Collapse — eval 154→20** | 100% crisis training with reward noise fix applied. Crisis games that fail to clear the wave are now skipped entirely (no survival bonus, no gradient). Only successful clears (+27 spike) contribute to training. Resumed from `polar2_pg_wave5_exploit_best.pth` (130 avg). Best eval 154.30 at iter 300, then steady collapse: 132→100→69→31→22 by iter 3500, flatlined at ~20–23 for remaining iters. Crisis train_avg kept climbing (30→60→80) — model learned to clear crisis waves but catastrophically forgot normal gameplay. Fixing the reward noise did not prevent collapse; the core problem is catastrophic forgetting from training exclusively on crisis scenarios. |
| `polar2_crisis_mix90` | polar2 | `polar2_crisis_mix90_policy_gradient.log` | *in progress* | 90% crisis + 10% normal (28 crisis + 4 normal per batch) with reward noise fix. Resumed from `polar2_pg_wave5_exploit_best.pth` (130 avg). Testing whether more normal games (4 vs 1) can anchor the policy against crisis drift. | Crisis games that fail to clear the wave are now skipped entirely (no survival bonus, no gradient). Only successful clears (+27 spike) contribute to training. Resumed from `polar2_pg_wave5_exploit_best.pth` (130 avg). Best eval 154.30 at iter 300, then steady collapse: 132→100→69→31→22 by iter 3500, flatlined at ~20–23 for remaining iters. Crisis train_avg kept climbing (30→60→80) — model learned to clear crisis waves but catastrophically forgot normal gameplay. Fixing the reward noise did not prevent collapse; the core problem is catastrophic forgetting from training exclusively on crisis scenarios. |

#### Crisis Mode Reward Signal Problem

Crisis mode disables per-asteroid scoring — the only score event is +27 for clearing the entire wave. Most crisis games end without clearing, so the per-frame reward is just the survival bonus (0.001). When every frame has the same tiny reward, the discounted returns form a near-flat curve with a tiny spread. Reward normalization then divides this small spread by a small standard deviation, producing full-magnitude "advantages" that are effectively random noise — early frames get positive values and late frames get negative values purely based on timing, not action quality. The model receives a full-strength gradient update pointing in a meaningless direction, causing weight drift that degrades the normal-game policy over time. The `polar2_crisis` and `polar2_crisis_mix97` runs were both affected by this bug — the steady collapse in `polar2_crisis` and the wild instability in `polar2_crisis_mix97` were caused by noise from failed crisis games polluting the gradient.

**Fix:** Remove the survival bonus for crisis games and skip any game with zero total reward. When the agent fails to clear a wave, the reward is truly zero everywhere — no normalization artifacts, no gradient, no noise. Only successful crisis clears (where the +27 spike creates real contrast between frames) contribute to training.

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

**Promote Checkpoint:**
```bash
tools/promote_checkpoint.sh          # most recent .pth
tools/promote_checkpoint.sh --best   # most recent *best*.pth
```
Copies the most recently modified `.pth` file from `nn_checkpoints/` to `nn_weights/`. Prints the destination path to stdout for piping.

**Benchmark:**
```bash
tools/benchmark.sh nn_weights/model.pth          # 1000 games (default)
tools/benchmark.sh nn_weights/model.pth 500      # 500 games
tools/promote_checkpoint.sh | tools/benchmark.sh  # promote then benchmark
```
Runs headless polar2 benchmarking games and reports scores.

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
