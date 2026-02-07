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
│       └── neural.py      # Neural network AI
│
├── training/              # Training scripts
│   ├── main_pg.py         # Policy gradient (REINFORCE) training
│   └── main_genetic.py    # Genetic algorithm optimization
│
├── tools/                 # Utility scripts
│   └── compact_recordings.py  # Consolidate game recordings
│
├── main_arcade.py         # Interactive game (keyboard/AI)
├── main_headless.py       # Headless benchmarking
```

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

# Watch neural network AI play (requires trained model)
python main_arcade.py --ain
python main_arcade.py --ain nn_weights/best_model.pth
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

# Benchmark neural network AI
python main_headless.py --ai-type neural --model-path nn_weights/model.pth -n 50

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
python training/main_pg.py --batch-size 32 --workers 8
```

**Optimize Heuristic AI (Genetic Algorithm):**
```bash
python training/main_genetic.py --population 50 --generations 20
```

#### Training Outputs

**Neural Network Training** (`training/main_pg.py`):

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

**Genetic Algorithm Training** (`training/main_genetic.py`):

Log file saved to root directory:

- **Log File**: `genetic.log`
  - Complete console output from optimization run
  - Includes generation progress and best parameters
  - Overwritten on each new run

### Utilities

**Consolidate Recording Files:**
```bash
python tools/compact_recordings.py
```

## AI Implementations

### Heuristic AI
Rule-based AI using:
- **Evasive maneuvers** - Weighted threat vectors from nearby asteroids
- **Speed control** - Target velocity management
- **Predictive targeting** - Intercept calculations for shooting

### Neural Network AI
Deep learning model trained with REINFORCE policy gradient:
- State: Player position/velocity, asteroid positions/velocities
- Actions: Turn left/right, accelerate/decelerate, shoot
- Reward: Game score with survival time bonus
