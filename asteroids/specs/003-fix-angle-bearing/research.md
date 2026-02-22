# Research: Fix Angle Normalization and Bearing Encoding

## R1: Angle Normalization Approach

**Decision**: Use `angle % (2 * math.pi)` after the angle update in `Player.update()`.

**Rationale**: Python's modulo operator with a positive divisor always returns a non-negative result, so `angle % (2*pi)` maps any real-valued angle to [0, 2pi). This is a single line change at the point where the angle is modified (game.py line 159).

**Alternatives considered**:
- `math.fmod()`: Returns values with the same sign as the dividend, so negative angles stay negative. Rejected.
- Normalizing in `compute_state` only: Leaves the bug in the game itself; other systems (rendering, heuristic AI) still see unbounded values. Rejected.

## R2: Bearing Encoding (cos/sin vs. Normalized Scalar)

**Decision**: Replace `angle / (2*pi)` with `(cos(angle), sin(angle))` as two separate state vector features.

**Rationale**: A unit vector bearing is:
- Continuous: no discontinuity at 0/2pi boundary
- Bounded: both components always in [-1, 1]
- Linearly decomposable: linear layers can directly compute angular relationships
- Standard practice in game AI and robotics

**Alternatives considered**:
- Keep scalar but normalize to [0, 1] after angle fix: Still has discontinuity at 0/1 boundary. Rejected.
- Use `(sin, cos)` order: Convention varies; `(cos, sin)` matches (x, y) component order which aligns with existing velocity encoding. Chosen.

## R3: Input Dimension Change Impact

**Decision**: Change from 141 to 142 inputs. Player state goes from 6 to 7 features (one angle replaced by two bearing components).

**Rationale**: The state vector layout changes at index 4:
- Old: `[x, y, vx, vy, angle_norm, cooldown, ast_0_x, ...]` (141 total)
- New: `[x, y, vx, vy, cos_angle, sin_angle, cooldown, ast_0_x, ...]` (142 total)

All downstream code that hardcodes 141 or the column layout must be updated.

**Files requiring changes**:
1. `asteroids/core/game.py:159` - Add angle normalization
2. `asteroids/ai/neural.py:41` - Change angle encoding to cos/sin
3. `asteroids/ai/neural.py:70` - Change `player_state_count = 6` to `7`
4. `tools/analyze_state_data.py:22,28` - Update column names and count

**Files that auto-adapt** (no changes needed):
- `training/cross_entropy.py` - DataLoader is dimension-agnostic
- `training/policy_gradient.py` - Uses `compute_state()`, no hardcoded dims
- `tools/compact_recordings.py` - Passes state arrays through unchanged
- `asteroids/core/game_runner.py` - Calls `compute_state()`, no hardcoded dims

## R4: Model Load Validation Strategy

**Decision**: Add a validation wrapper function that checks loaded state dict shapes against model before calling `load_state_dict`.

**Rationale**: PyTorch's `load_state_dict(strict=True)` already throws an error on shape mismatch, but the error message is generic (mentions tensor sizes). A pre-check can provide a domain-specific message like "Model was trained with 141 inputs but current architecture expects 142 inputs. The model file is incompatible with the current code."

**Locations needing validation** (4 sites):
1. `main_arcade.py:138` - Interactive play
2. `main_headless.py:197` - Headless benchmarking
3. `training/cross_entropy.py:171` - Training evaluation
4. `training/policy_gradient.py:126` - Policy gradient workers

**Approach**: Add a `validate_and_load_model` function to `asteroids/ai/neural.py` that all 4 call sites use.

## R5: Training Data Conversion

**Decision**: Create `tools/convert_training_data.py` following the same patterns as existing tools.

**Rationale**: The existing tools (`analyze_state_data.py`, `compact_recordings.py`) establish conventions:
- Files in `tools/` directory
- Use `argparse` for CLI arguments
- Pattern `data/<base_name>_*.npz` for file discovery
- Sort files numerically by suffix

**Conversion logic**:
1. Load each `.npz` file
2. Extract column 4 (normalized angle = `angle / (2*pi)`)
3. Compute `angle_radians = col4 * 2 * pi`
4. Compute `cos_col = cos(angle_radians)` and `sin_col = sin(angle_radians)`
5. Replace column 4 with `cos_col`, insert `sin_col` at column 5
6. Save with new base name, preserving numeric suffix
