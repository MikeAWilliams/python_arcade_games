# Feature Specification: Fix Angle Normalization and Bearing Encoding

**Feature Branch**: `003-fix-angle-bearing`
**Created**: 2026-02-21
**Status**: Draft
**Input**: User description: "Fix unbounded player angle bug, replace scalar angle with unit vector bearing in NN model, add parameter count validation on model load, and create a training data conversion tool."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Bounded Player Angle in Game (Priority: P1)

As a player or AI, the ship's angle should always remain within [0, 2pi) so that the same physical heading always produces the same numerical value.

**Why this priority**: This is the root bug. The unbounded angle accumulates over time, causing the same physical heading to produce wildly different values (e.g., 0, 6pi, -4pi all mean "facing right"). This corrupts all downstream systems including the NN state encoding. Fixing this first ensures all other changes build on correct data.

**Independent Test**: Run a game (interactive or headless) where the ship spins continuously for 60+ seconds. Inspect the player angle value; it must always be in [0, 2pi).

**Acceptance Scenarios**:

1. **Given** a player at angle pi/2, **When** the player turns left continuously for 100 seconds at PLAYER_TURN_RATE, **Then** the angle is always in [0, 2pi) at every tick.
2. **Given** a player at angle pi/2, **When** the player turns right continuously for 100 seconds, **Then** the angle is always in [0, 2pi) at every tick.
3. **Given** a new wave starts, **When** the player resets, **Then** the angle is pi/2 (facing up).

---

### User Story 2 - Unit Vector Bearing in NN State (Priority: P1)

As a developer training neural networks, the player heading should be encoded as a unit vector (cos(theta), sin(theta)) instead of a scalar angle. This changes the NN input dimension from 141 to 142.

**Why this priority**: Equal to P1 because this change is tightly coupled to the angle fix. The scalar angle (even when bounded) has a discontinuity at 0/2pi. A unit vector bearing is continuous, bounded to [-1, 1], and directly usable by linear layers.

**Independent Test**: Run a game with the neural AI. Inspect the state vector; index 4 should be cos(theta) and index 5 should be sin(theta), both in [-1, 1]. The total state vector length should be 142.

**Acceptance Scenarios**:

1. **Given** a player at angle pi/2, **When** compute_state is called, **Then** the bearing values at indices 4 and 5 are (cos(pi/2), sin(pi/2)) = (0.0, 1.0).
2. **Given** a player at angle 0, **When** compute_state is called, **Then** the bearing values are (1.0, 0.0).
3. **Given** the NN model is constructed, **When** num_inputs is checked, **Then** it equals 142 (was 141).

---

### User Story 3 - Model Parameter Validation on Load (Priority: P2)

As a developer, when I load a saved model (.pth file), the system should immediately error if the parameter dimensions don't match the current model architecture, rather than silently producing wrong results or crashing later.

**Why this priority**: This is a safety net. The input dimension change from 141 to 142 invalidates all existing saved models. Without validation, loading an old model would either crash cryptically or silently produce garbage. This provides a clear, immediate error message.

**Independent Test**: Attempt to load an old 141-input model file with the new 142-input architecture. The system should raise an error with a clear message about the dimension mismatch.

**Acceptance Scenarios**:

1. **Given** a saved model with 141 inputs, **When** the system attempts to load it into a 142-input architecture, **Then** the system raises an error with a message indicating the expected vs actual parameter count.
2. **Given** a saved model with 142 inputs (correct), **When** the system loads it, **Then** loading succeeds without error.
3. **Given** parameter validation fails, **When** the error is reported, **Then** the message clearly states which dimensions mismatched and suggests that the model file is incompatible.

---

### User Story 4 - Training Data Conversion Tool (Priority: P2)

As a developer, I need to convert my existing training data (which contains the old normalized unbounded angle at column index 4) to the new bearing format (cos, sin at columns 4-5) so I can retrain on corrected data.

**Why this priority**: Without this tool, ~58 GB of training data is unusable. The tool restores the value of existing data by converting it to the new format.

**Independent Test**: Run the conversion tool on `test_data` with a new output base name. Load the converted file and verify column 4 contains cos values in [-1, 1], column 5 contains sin values in [-1, 1], and the state vector width is 142 (was 141).

**Acceptance Scenarios**:

1. **Given** a data file with base name "test_data" containing old-format states (141 columns), **When** the tool is run with `--input-base test_data --output-base test_data_v2`, **Then** new files are created in data/ with base name "test_data_v2" and each has 142-column states.
2. **Given** old data where column 4 contains `angle / (2 * pi)`, **When** the tool converts it, **Then** column 4 becomes `cos(angle)` and a new column 5 becomes `sin(angle)` where `angle = old_column_4 * 2 * pi`.
3. **Given** multiple files matching the input base name pattern, **When** the tool runs, **Then** all matching files are converted and numbered consistently.
4. **Given** no files match the input base name, **When** the tool runs, **Then** it reports a clear error.

---

### Edge Cases

- What happens when the angle is exactly 2pi? It should wrap to 0.
- What happens when the angle is negative (pre-fix data)? The conversion tool should handle any real-valued angle by computing cos/sin directly (which is valid for all real numbers).
- What happens if a data file is corrupted or has unexpected column count? The conversion tool should report which file failed and continue processing remaining files.
- What happens if the output base name already has existing files? The tool should overwrite them (standard behavior for data conversion scripts).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The game MUST normalize the player angle to [0, 2pi) after every angle update in the Player.update method.
- **FR-002**: The NN state encoder (`compute_state`) MUST encode the player heading as two values: cos(theta) and sin(theta), replacing the single normalized angle value.
- **FR-003**: The NN model input dimension MUST be 142 (6 player features now include bearing_x, bearing_y instead of angle, plus 135 asteroid features).
- **FR-004**: The player_state_count in NNAIParameters MUST be updated from 6 to 7 (x, y, vx, vy, bearing_x, bearing_y, shot_cooldown).
- **FR-005**: Every location that loads a model from a .pth file MUST validate that the loaded state dict dimensions match the current model architecture before use, and raise a clear error on mismatch.
- **FR-006**: A conversion tool MUST exist at `tools/convert_training_data.py` that reads old-format training data files, converts the angle encoding from scalar to unit vector bearing, and writes new-format files.
- **FR-007**: The conversion tool MUST accept command-line arguments for input base name and output base name.
- **FR-008**: The conversion tool MUST find all files matching `data/<input_base>_*.npz`, process each one, and write corresponding `data/<output_base>_*.npz` files preserving the numeric suffix.
- **FR-009**: The conversion tool MUST convert column 4 (normalized angle) by: computing `angle_radians = old_value * 2 * pi`, then replacing it with `cos(angle_radians)` and inserting `sin(angle_radians)` as a new column, shifting subsequent columns right.
- **FR-010**: The conversion tool MUST preserve all other data arrays (actions, game_ids, tick_nums) unchanged.

### Key Entities

- **State Vector**: The numerical representation of game state fed to the NN. Changes from 141 to 142 dimensions. Player heading changes from `[angle/(2pi)]` to `[cos(angle), sin(angle)]`.
- **Player Angle**: The ship's facing direction in radians. Currently unbounded; must be normalized to [0, 2pi).
- **Training Data File**: NPZ archive containing `states`, `actions`, `game_ids`, `tick_nums` arrays. The `states` array column layout changes at index 4.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Player angle never exceeds [0, 2pi) during any game session of any length.
- **SC-002**: The NN state vector is 142 dimensions and the bearing values (indices 4, 5) are always in [-1, 1].
- **SC-003**: Loading an incompatible model file produces a clear error message within 1 second rather than silently failing or crashing later.
- **SC-004**: The conversion tool successfully converts all training data files, producing output files where each state vector has 142 columns with correct bearing values.
- **SC-005**: A newly trained model using converted data can run in both arcade and headless modes without errors.

## Assumptions

- The angle normalization uses modulo 2pi (i.e., `angle % (2 * math.pi)`), which keeps values in [0, 2pi).
- The conversion formula `angle_radians = normalized_angle * 2 * pi` correctly recovers the original angle from old training data, regardless of whether that angle was bounded or not, because cos/sin are periodic and will produce the correct bearing for any real-valued input.
- Existing model weight files (.pth) in nn_weights/ and nn_checkpoints/ are incompatible with the new architecture and should not be deleted (user may want to keep them for reference).
- The conversion tool is placed in the existing `tools/` directory alongside other data utility scripts.
- PyTorch's `load_state_dict` with `strict=True` (the default) already catches shape mismatches, but the requirement is to add an explicit, user-friendly validation with a clear error message before the load attempt.
