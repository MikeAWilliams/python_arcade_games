# Tasks: Fix Angle Normalization and Bearing Encoding

**Input**: Design documents from `asteroids/specs/003-fix-angle-bearing/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md

**Tests**: No tests requested. This project uses manual verification (interactive play, headless benchmarking, analyze_state_data.py).

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup

**Purpose**: No setup needed. This feature modifies existing files in an established project. No new dependencies, no new directories.

_(No tasks in this phase)_

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: The angle normalization fix in game.py is the root change that all other work depends on. It must be done first so that new game recordings and live state vectors have correct, bounded angles.

- [x] T001 [US1] Add angle normalization (`self.geometry.angle %= 2 * math.pi`) after the angle update on line 159 of `asteroids/core/game.py` in the `Player.update()` method

**Checkpoint**: Player angle is now always in [0, 2pi). Verify by running `python main_arcade.py` and observing the ship can spin freely without angle growing unbounded.

---

## Phase 3: User Story 2 - Unit Vector Bearing in NN State (Priority: P1)

**Goal**: Replace the scalar angle in the NN state vector with a (cos, sin) unit vector bearing, changing input dimension from 141 to 142.

**Independent Test**: Run `python main_arcade.py --ais` (heuristic AI) to confirm the game works with the angle fix. Then inspect `compute_state` output to verify bearing values at indices 4-5 are in [-1, 1] and state vector length is 142.

### Implementation

- [x] T002 [US2] In `asteroids/ai/neural.py`, in `compute_state()` function (line 41), replace `result.append(float(game.player.geometry.angle / (2 * math.pi)))` with two lines: `result.append(float(math.cos(game.player.geometry.angle)))` and `result.append(float(math.sin(game.player.geometry.angle)))`. Update the docstring (lines 22-25) to reflect the new state layout: `player_state: x, y, vx, vy, bearing_x, bearing_y, shot_cooldown`.
- [x] T003 [US2] In `asteroids/ai/neural.py`, in `NNAIParameters.__init__()` (line 70), change `player_state_count = 6` to `player_state_count = 7` and update the comment to `# x,y, vx,vy, bearing_x,bearing_y, shot_cooldown`.
- [x] T004 [P] [US2] In `tools/analyze_state_data.py`, in `build_column_names()` (line 28), replace `cols.append("player_angle (norm)")` with `cols.append("player_bearing_x (cos)")` and `cols.append("player_bearing_y (sin)")`. Update the docstring on line 22 from "141" to "142".

**Checkpoint**: NN model now has 142 inputs. The `compute_state` function returns 142 values. The existing assertion in `NNAIInputMethod.compute_state()` (line 108) will confirm the match. Verify by running `python main_headless.py -n 1 --ai-type neural` (will use a fresh randomly-initialized model).

---

## Phase 4: User Story 3 - Model Parameter Validation on Load (Priority: P2)

**Goal**: Add a validation helper that checks model weight dimensions before loading, then use it at all 4 call sites so incompatible models produce a clear error.

**Independent Test**: Try loading an old model file (e.g., `nn_weights/nn_model.pth` which has 141 inputs) with the new architecture. Should get a clear error message about dimension mismatch.

### Implementation

- [x] T005 [US3] In `asteroids/ai/neural.py`, add a new function `validate_and_load_model(model, state_dict, source_description="model file")` that compares `model.state_dict()["0.weight"].shape` with `state_dict["0.weight"].shape`, raises `ValueError` with a clear message on mismatch, and calls `model.load_state_dict(state_dict)` on success.
- [x] T006 [P] [US3] In `main_arcade.py` (line 138), import `validate_and_load_model` from `asteroids.ai.neural` and replace the bare `params.model.load_state_dict(torch.load(...))` call with `validate_and_load_model(params.model, torch.load("nn_weights/" + args.ain, map_location=device), source_description=args.ain)`.
- [x] T007 [P] [US3] In `main_headless.py` (lines 197-199), import `validate_and_load_model` from `asteroids.ai.neural` and replace the bare `params.model.load_state_dict(torch.load(...))` call with `validate_and_load_model(params.model, torch.load("nn_weights/" + model_path, map_location="cpu"), source_description=model_path)`.
- [x] T008 [P] [US3] In `training/cross_entropy.py` (line 171), import `validate_and_load_model` from `asteroids.ai.neural` and replace `eval_params.model.load_state_dict(model_state_dict)` with `validate_and_load_model(eval_params.model, model_state_dict, source_description="training checkpoint")`.
- [x] T009 [P] [US3] In `training/policy_gradient.py` (line 126), import `validate_and_load_model` from `asteroids.ai.neural` and replace `params.model.load_state_dict(model_state_dict)` with `validate_and_load_model(params.model, model_state_dict, source_description="training checkpoint")`.

**Checkpoint**: Old model files now produce clear errors. New models (trained after this change) load successfully. Verify by running `python main_arcade.py --ain nn_model.pth` and confirming a clear error about 141 vs 142 inputs.

---

## Phase 5: User Story 4 - Training Data Conversion Tool (Priority: P2)

**Goal**: Create a script that converts old-format training data (141-column states with scalar angle) to new-format (142-column states with cos/sin bearing).

**Independent Test**: Run `python tools/convert_training_data.py --input-base test_data --output-base test_data_v2`, then run `python tools/analyze_state_data.py --base-name test_data_v2` and verify columns 4-5 are in [-1, 1] and state width is 142.

### Implementation

- [x] T010 [US4] Create `tools/convert_training_data.py` with: argparse CLI accepting `--input-base` and `--output-base` arguments; file discovery using `data/<input_base>_*.npz` pattern sorted numerically by suffix; for each file: load states array, extract column 4 (normalized angle), compute `angle_rad = col4 * 2 * pi`, replace column 4 with `cos(angle_rad)`, insert `sin(angle_rad)` as new column 5 (shifting remaining columns right) to produce 142-column states, save as `data/<output_base>_<suffix>.npz` preserving actions/game_ids/tick_nums unchanged. Print progress for each file converted. Error clearly if no files match the input pattern.

**Checkpoint**: Conversion tool works on test data. Verify with `python tools/analyze_state_data.py --base-name test_data_v2`.

---

## Phase 6: Polish & Cross-Cutting Concerns

- [x] T011 Run `./format.sh` to apply Black formatting to all modified files
- [x] T012 Verify end-to-end: run `python main_arcade.py --ais` to confirm interactive gameplay works with angle fix
- [x] T013 Verify end-to-end: run `python main_headless.py -n 5 --ai-type neural` to confirm headless mode works with new 142-input model

---

## Dependencies & Execution Order

### Phase Dependencies

- **Foundational (Phase 2)**: No dependencies - T001 is the root fix
- **User Story 2 (Phase 3)**: Depends on T001 (angle must be bounded before encoding it)
- **User Story 3 (Phase 4)**: Depends on T002-T003 (model dim must be 142 before validation makes sense)
- **User Story 4 (Phase 5)**: Independent of T002-T009 (converts old data regardless of current code state), but logically follows Phase 3
- **Polish (Phase 6)**: Depends on all previous phases

### Within Each User Story

- **US2**: T002 and T003 must be done together (both in neural.py, tightly coupled). T004 is independent [P].
- **US3**: T005 must come first (defines the helper). T006-T009 are all independent [P] (different files).
- **US4**: Single task T010.

### Parallel Opportunities

- T004 can run in parallel with T002/T003 (different files)
- T006, T007, T008, T009 can all run in parallel (4 different files, all depend only on T005)

---

## Parallel Example: User Story 3

```text
# After T005 completes, launch all call-site updates in parallel:
Task: T006 - Update main_arcade.py to use validate_and_load_model
Task: T007 - Update main_headless.py to use validate_and_load_model
Task: T008 - Update training/cross_entropy.py to use validate_and_load_model
Task: T009 - Update training/policy_gradient.py to use validate_and_load_model
```

---

## Implementation Strategy

### MVP First (US1 + US2)

1. Complete T001: Fix angle normalization (the bug fix)
2. Complete T002-T004: Bearing encoding (the encoding improvement)
3. **STOP and VALIDATE**: Game works, state vector is 142 dims, bearing values bounded
4. This alone fixes the core problem

### Incremental Delivery

1. T001 → Angle bug fixed
2. T002-T004 → NN encoding fixed, model architecture updated
3. T005-T009 → Safety net for loading old models
4. T010 → Old training data is recoverable
5. T011-T013 → Everything formatted and verified

---

## Notes

- US1 (angle fix) is a single line in game.py - it's the foundation for everything else
- US2 (bearing encoding) is tightly coupled with US1 and should be done immediately after
- All existing .pth model files become incompatible after US2 - this is expected and US3 makes it fail gracefully
- The ~58 GB of training data can be converted with US4 tool but this is a long-running operation the user will run manually
- No new dependencies needed - only uses existing numpy, torch, math, argparse
