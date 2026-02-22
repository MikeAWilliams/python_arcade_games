# Implementation Plan: Fix Angle Normalization and Bearing Encoding

**Branch**: `003-fix-angle-bearing` | **Date**: 2026-02-21 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `asteroids/specs/003-fix-angle-bearing/spec.md`

## Summary

Fix the unbounded player angle bug by normalizing to [0, 2pi), replace the scalar angle in the NN state vector with a unit vector bearing (cos, sin), add model parameter validation on load, and create a tool to convert existing training data.

## Technical Context

**Language/Version**: Python 3.12
**Primary Dependencies**: arcade 3.3.3, PyTorch, NumPy
**Storage**: NPZ files for training data, .pth files for model weights
**Testing**: Manual - run game interactively and headless; verify with analyze_state_data.py
**Target Platform**: Linux (local development)
**Project Type**: Single project (monorepo sub-project at `asteroids/`)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
| --------- | ------ | ----- |
| I. Learning Focus | PASS | Bug fix + encoding improvement directly supports AI learning |
| II. Code Simplicity | PASS | All changes are minimal and direct (1-line angle fix, 2-line encoding change, simple validation function, straightforward conversion script) |
| III. Source-Only Execution | PASS | No packaging changes |
| IV. Documentation Balance | PASS | Only updating comments that describe column layout |
| V. Permission-First | PASS | Will ask before file/git operations |
| VI. User-Written AI Algorithms | PASS | This is infrastructure/bug-fix work, not AI algorithm design. The user designed the bearing encoding approach. |

No violations. No complexity tracking needed.

## Project Structure

### Documentation (this feature)

```text
asteroids/specs/003-fix-angle-bearing/
├── spec.md
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
└── checklists/
    └── requirements.md
```

### Source Code (files to modify)

```text
asteroids/
├── asteroids/
│   ├── ai/
│   │   └── neural.py          # Bearing encoding, input dim, validation helper
│   └── core/
│       └── game.py            # Angle normalization
├── main_arcade.py             # Use validation helper
├── main_headless.py           # Use validation helper
├── training/
│   ├── cross_entropy.py       # Use validation helper
│   └── policy_gradient.py     # Use validation helper
└── tools/
    ├── analyze_state_data.py  # Update column names
    └── convert_training_data.py  # NEW - data conversion tool
```

**Structure Decision**: No new directories needed. The conversion tool goes in the existing `tools/` directory following established conventions. All other changes modify existing files.

## Implementation Details

### Change 1: Angle Normalization (game.py)

**File**: `asteroids/core/game.py`, line 159

Add angle normalization after the angle update in `Player.update()`:

```python
# Current:
self.geometry.angle += self.angle_vel * dt

# New:
self.geometry.angle += self.angle_vel * dt
self.geometry.angle %= 2 * math.pi
```

Single line addition. No other angle-related code needs to change because `cos()` and `sin()` already work correctly with any angle value.

### Change 2: Bearing Encoding (neural.py)

**File**: `asteroids/ai/neural.py`

**a) `compute_state()` function (line 41):**

Replace:
```python
result.append(float(game.player.geometry.angle / (2 * math.pi)))
```

With:
```python
result.append(float(math.cos(game.player.geometry.angle)))
result.append(float(math.sin(game.player.geometry.angle)))
```

Update docstring to reflect new layout.

**b) `NNAIParameters.__init__()` (line 70):**

Change:
```python
player_state_count = 6  # x,y, vx,vy,theta, shot_cooldown
```

To:
```python
player_state_count = 7  # x,y, vx,vy, bearing_x,bearing_y, shot_cooldown
```

### Change 3: Model Validation Helper (neural.py)

Add a function to `asteroids/ai/neural.py`:

```python
def validate_and_load_model(model, state_dict, source_description="model file"):
    """Load state dict into model with dimension validation."""
    # Check first layer input dimension
    expected_shape = model.state_dict()["0.weight"].shape
    loaded_shape = state_dict["0.weight"].shape
    if expected_shape != loaded_shape:
        raise ValueError(
            f"Model incompatible: first layer expects {expected_shape[1]} inputs "
            f"but {source_description} has {loaded_shape[1]}. "
            f"The model file was likely trained with a different architecture."
        )
    model.load_state_dict(state_dict)
```

Replace bare `load_state_dict` calls at all 4 sites with calls to this function.

### Change 4: Update Call Sites

**main_arcade.py (line 138):**
```python
# Import validate_and_load_model, then:
validate_and_load_model(params.model, torch.load(...), source_description=args.ain)
```

**main_headless.py (line 197):**
```python
validate_and_load_model(params.model, torch.load(...), source_description=model_path)
```

**training/cross_entropy.py (line 171):**
```python
validate_and_load_model(eval_params.model, model_state_dict, source_description="training checkpoint")
```

**training/policy_gradient.py (line 126):**
```python
validate_and_load_model(params.model, model_state_dict, source_description="training checkpoint")
```

### Change 5: Update Column Names (analyze_state_data.py)

**File**: `tools/analyze_state_data.py`, lines 22-28

Replace:
```python
cols.append("player_angle (norm)")
```

With:
```python
cols.append("player_bearing_x (cos)")
cols.append("player_bearing_y (sin)")
```

Update docstring comment from "141" to "142".

### Change 6: Conversion Tool (NEW tools/convert_training_data.py)

New file following the pattern of existing tools. Key logic:

```python
# For each file matching data/<input_base>_*.npz:
raw = np.load(file)
states = raw["states"]  # shape (N, 141)

# Extract old angle column and convert
angle_norm = states[:, 4]            # angle / (2*pi)
angle_rad = angle_norm * 2 * np.pi   # recover radians
cos_col = np.cos(angle_rad)
sin_col = np.sin(angle_rad)

# Build new states: cols 0-3, cos, sin, cols 5-140
new_states = np.concatenate([
    states[:, :4],
    cos_col.reshape(-1, 1),
    sin_col.reshape(-1, 1),
    states[:, 5:]
], axis=1)  # shape (N, 142)

# Save with new base name, same suffix
np.savez_compressed(output_file,
    states=new_states,
    actions=raw["actions"],
    game_ids=raw["game_ids"],
    tick_nums=raw["tick_nums"])
```

CLI: `python tools/convert_training_data.py --input-base <name> --output-base <name>`

## Complexity Tracking

No constitution violations. No complexity justifications needed.
