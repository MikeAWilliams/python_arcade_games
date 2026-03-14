# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An Asteroids game implementation in Python with pluggable AI input system. The project supports both interactive gameplay and headless multi-process AI benchmarking.

## Running the Game

### Interactive Mode (Arcade)
```bash
python main_arcade.py          # Keyboard control
python main_arcade.py --aih    # Heuristic AI
python main_arcade.py --air    # RawGeometry Neural Network AI (default: nn_model.pth)
python main_arcade.py --aip    # Polar Neural Network AI (default: nn_polar.pth)
```

### Headless Benchmarking Mode
```bash
python main_headless.py -n 10                          # Run 10 games on all CPU cores
python main_headless.py -n 100 -t 8                    # 100 games on 8 cores
python main_headless.py --ai-type heuristic --seed 42  # Reproducible heuristic AI runs
python main_headless.py --ai-type raw                  # RawGeometry NN (default: nn_model.pth)
python main_headless.py --ai-type polar                # Polar NN (default: nn_polar.pth)
```

Headless mode uses `ProcessPoolExecutor` to bypass Python's GIL and run multiple game instances in true parallel across CPU cores.

## Development Commands

### Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # Only dependency: arcade==3.3.3
```

### Code Quality
```bash
./format.sh      # Black formatting (MUST run after changes)
black *.py       # Alternative formatting command
```

## Architecture

### Core Files
- **asteroids/core/game.py** - Game logic, entities (Player, Asteroid, Bullet), physics, collision detection
- **asteroids/core/game_runner.py** - Shared game execution library, AI factory
- **main_arcade.py** - Interactive entry point with Arcade rendering
- **main_headless.py** - Multi-process headless benchmarking
- **asteroids/ai/heuristic.py** - Heuristic AI implementation
- **asteroids/ai/raw_geometry_nn.py** - Neural network AI with raw geometry state (142 inputs)
- **asteroids/ai/polar_nn.py** - Neural network AI with polar-coordinate features (39 inputs)

### Strategy Pattern for Input

All input methods implement the `InputMethod` abstract base class:

```python
class InputMethod(ABC):
    @abstractmethod
    def get_move(self) -> Action:
        pass
```

Implementations:
- **KeyboardInput** - User keyboard control
- **HeuristicAIInput** - Intelligent heuristic AI with threat vectors and predictive targeting
- **RawGeometryNNInputMethod** - Neural network fed raw game geometry (player + asteroid positions/velocities)
- **PolarNNInputMethod** - Neural network fed pre-computed polar features (distance, relative angle, closing speed per asteroid)

### Game Loop Pattern

Fixed timestep update-render cycle:
1. Clear turn/acceleration state
2. Get action from input method (`get_move()`)
3. Execute action on game state
4. Update physics (dt = 1/60 second)
5. Render (arcade mode) or continue (headless mode)

### Entity Management

All entities follow consistent pattern:
- `GeometryObject` - Position (`Vec2d`), radius, angle
- `vel` property - Velocity as `Vec2d`
- `update(dt)` method - Physics update
- `copy_geometry()` method - Returns immutable copy for rendering

### Separation of Game Logic and Rendering

`GameView` class in `main_arcade.py` separates concerns:
- Requests immutable geometry state via `game.geometry_state()`
- Rendering never mutates game state
- Returns copies of all entities to prevent accidental modification

### Progressive Difficulty System

Wave-based difficulty scaling:
- Speed multiplier increases by 1.1x each wave
- Player resets to center with zero velocity
- 3 new large asteroids spawn each wave
- Asteroid splitting: Big → 2 Medium → 3 Small

## Code Style (Enforced by Black)

### Naming Conventions
- Classes: PascalCase (`GameView`, `SmartAIInput`)
- Functions/Variables: snake_case (`get_move`, `player_pos`)
- Constants: UPPER_SNAKE_CASE (`PLAYER_TURN_RATE`, `BULLET_SPEED`)

### Import Organization
1. Standard library (math, random, sys, etc.)
2. Third-party (arcade)
3. Local (from game import ...)

### Type Hints
Use type hints for all function signatures and class attributes.

## Key Constants (game.py)

```python
PLAYER_TURN_RATE = math.pi         # Radians per second
PLAYER_ACCELERATION = 180           # Pixels/sec²
BULLET_SPEED = 500                  # Pixels/sec
ASTEROID_BASE_SPEED = 100           # Pixels/sec
ASTEROID_SPEED_INCREMENT = 1.1      # Wave multiplier
SHOOT_COOLDOWN = 20                 # Frames
```

## AI Development Pattern

### Neural Network Models

Two NN architectures:
- **RawGeometryNN** (142 inputs): Feeds full game geometry directly. Architecture: 142 → 128 (ReLU) → 6.
- **PolarNN** (39 inputs): Pre-computes player-relative features. Architecture: 39 → 128 (ReLU) → 64 (ReLU) → 6. Asteroids sorted by distance, top 9 kept.

PolarNN state vector:
- Global (3): player speed, shot cooldown, asteroid count
- Per asteroid (4 × 9): distance, relative angle from bearing, closing speed, size category

### Tools
- `tools/generate_random_model.py` - Generate random model weights (`--model raw` or `--model polar`)
- `tools/visualize_state.py` - Visualize raw geometry state recordings
- `tools/visualize_state_polar.py` - Visualize polar state (converted from raw recordings)
- `tools/analyze_state_data.py` - Analyze training data statistics

### Training
- `training/cross_entropy.py` - Supervised learning from recorded heuristic AI data
- `training/policy_gradient.py` - REINFORCE policy gradient training

To create new AI:
1. Implement `InputMethod` interface
2. Override `get_move()` to return appropriate `Action`
3. Register in `game_runner.py:create_ai_input()` and entry points
4. Use headless mode for performance benchmarking

## Physics and Collision

- Simple Euclidean distance-based collision detection (`math.dist`)
- Bounce mechanics at screen boundaries
- Bullet pruning for off-screen projectiles
- No rotation physics - player instantly rotates at fixed rate

## Performance Considerations

- Use `copy_geometry()` for immutable state snapshots
- Prune off-screen bullets to avoid unbounded growth
- Headless mode bypasses rendering entirely for maximum throughput
- Multi-process architecture avoids Python GIL limitations
