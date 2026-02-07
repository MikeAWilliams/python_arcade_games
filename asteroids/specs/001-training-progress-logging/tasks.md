# Tasks: Training Progress Tracking and Logging

**Input**: Design documents from `/specs/001-training-progress-logging/`
**Prerequisites**: plan.md (required), spec.md (required for user stories)

**Tests**: No tests requested - empirical validation through running training scripts

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US5)
- Include exact file paths in descriptions

## Path Conventions

This is a single-project repository with training scripts in `training/` directory. All modifications are to existing files, no new files created.

---

## Phase 1: Setup (No setup needed)

**Purpose**: Project already initialized - modifying existing training scripts only

No setup tasks required. Proceeding directly to user story implementation.

---

## Phase 2: User Story 1 & 2 - Neural Network Training Logging (Priority: P1) 🎯 MVP

**Goal**: Enable dual logging (screen + file) for neural network training with timestamped logs in `nn_checkpoints/`

**Independent Test**: Run `python training/main_pg.py --batch-size 2 --workers 2` for a few epochs, verify console output appears AND log file is created in `nn_checkpoints/` with matching content.

**Why Combined**: US1 (real-time) and US2 (log file) are implemented together - same logging setup provides both capabilities.

### Implementation for User Stories 1 & 2

- [ ] T001 [US1+US2] Add imports (logging, datetime, pathlib) to training/main_pg.py
- [ ] T002 [US1+US2] Create setup_logging() function in training/main_pg.py to configure dual output (console + file in nn_checkpoints/)
- [ ] T003 [US1+US2] Call setup_logging() at start of train_model() to get logger and timestamp in training/main_pg.py
- [ ] T004 [US1+US2] Replace print() on line 196 (device info) with logger.info() in training/main_pg.py
- [ ] T005 [US1+US2] Replace print() on line 201 (worker count) with logger.info() in training/main_pg.py
- [ ] T006 [US1+US2] Replace print() on line 202 (batch size) with logger.info() in training/main_pg.py
- [ ] T007 [US1+US2] Replace print() on line 215 (worker pool creation) with logger.info() in training/main_pg.py
- [ ] T008 [US1+US2] Replace print() on lines 256-260 (epoch progress) with logger.info() in training/main_pg.py
- [ ] T009 [US1+US2] Replace print() on line 266 (training completion) with logger.info() in training/main_pg.py
- [ ] T010 [US1+US2] Replace print() on line 267 (model saved message) with logger.info() in training/main_pg.py
- [ ] T011 [US1+US2] Update checkpoint save path on line 242 to use nn_checkpoints/ with timestamp in training/main_pg.py
- [ ] T012 [US1+US2] Update final model save on line 263 to save to nn_checkpoints/checkpoint_{timestamp}_final.pth in training/main_pg.py

**Checkpoint**: Neural network training now logs to both console and file. Run test to verify.

**Test Steps**:
1. Run: `python training/main_pg.py --batch-size 2 --workers 2`
2. Let run for ~10 epochs, then Ctrl+C
3. Verify: Console shows output
4. Verify: Log file exists in nn_checkpoints/ with format training_YYYYMMDD_HHMMSS.log
5. Verify: Log file content matches console output
6. Verify: nn_checkpoints/ directory was created automatically

---

## Phase 3: User Story 3 - Checkpoint Management (Priority: P2)

**Goal**: Save full training state in organized checkpoint files (periodic, best, final) to enable future resume capability

**Independent Test**: Run training for multiple epochs, verify checkpoint files appear in `nn_checkpoints/` with proper naming and can be loaded with torch.load()

### Implementation for User Story 3

- [ ] T013 [US3] Add best model checkpoint save when avg_score > max_score in training/main_pg.py (save to nn_checkpoints/checkpoint_{timestamp}_best.pth)
- [ ] T014 [US3] Update checkpoint dict to include full state (model_state_dict, optimizer_state_dict, epoch, max_score, loss) in training/main_pg.py
- [ ] T015 [US3] Update periodic checkpoint save (line 241-242) to use new checkpoint format with full state in training/main_pg.py
- [ ] T016 [US3] Update final checkpoint save (added in T012) to use full state format in training/main_pg.py

**Checkpoint**: All checkpoints now save complete training state for potential resume.

**Test Steps**:
1. Run: `python training/main_pg.py --batch-size 2 --workers 2`
2. Let run through at least 6000 epochs (one checkpoint save)
3. Verify: Checkpoint file created in nn_checkpoints/
4. Verify: Filename format is checkpoint_YYYYMMDD_HHMMSS_epoch_6000.pth
5. Verify: Can load checkpoint: `torch.load('nn_checkpoints/checkpoint_*.pth')`
6. Verify: Checkpoint contains keys: epoch, model_state_dict, optimizer_state_dict, max_score, loss
7. Watch for best score improvement and verify best checkpoint is saved
8. Verify best checkpoint has higher score than later checkpoints if overfitting occurs

---

## Phase 4: User Story 5 - Genetic Algorithm Logging (Priority: P1)

**Goal**: Enable dual logging (screen + file) for genetic algorithm training with simple genetic.log in root directory

**Independent Test**: Run `python training/main_genetic.py --population 10 --generations 3`, verify console output appears AND genetic.log is created in root with matching content

### Implementation for User Story 5

- [ ] T017 [P] [US5] Add logging import to training/main_genetic.py
- [ ] T018 [US5] Create setup_genetic_logging() function in training/main_genetic.py to configure dual output (console + genetic.log in root, overwrite mode)
- [ ] T019 [US5] Call setup_genetic_logging() at start of run() method in GeneticAlgorithm class in training/main_genetic.py
- [ ] T020 [US5] Store logger as instance variable self.logger in training/main_genetic.py
- [ ] T021 [US5] Replace print() on lines 237-239 (algorithm config) with self.logger.info() in training/main_genetic.py
- [ ] T022 [US5] Replace print() on line 240 (empty line) with self.logger.info('') in training/main_genetic.py
- [ ] T023 [US5] Replace print() on line 315 (generation progress) with self.logger.info() in training/main_genetic.py
- [ ] T024 [US5] Replace print() on lines 352-376 (final results) with logger.info() in main() function of training/main_genetic.py

**Checkpoint**: Genetic algorithm now logs to both console and genetic.log file.

**Test Steps**:
1. Run: `python training/main_genetic.py --population 10 --generations 3`
2. Let complete
3. Verify: Console shows generation progress
4. Verify: genetic.log created in root directory (not in nn_checkpoints/)
5. Verify: genetic.log contains all console output (headers, generation stats, final results)
6. Run again: `python training/main_genetic.py --population 10 --generations 2`
7. Verify: genetic.log contains only second run's output (overwritten, not appended)

---

## Phase 5: Polish & Documentation

**Goal**: Format code and update documentation

### Polish Tasks

- [ ] T025 Run ./format.sh to format all Python files with Black
- [ ] T026 Update README.md Training section with neural network training outputs documentation
- [ ] T027 Update README.md Training section with genetic algorithm training outputs documentation

**Test Steps**:
1. Run: `./format.sh`
2. Verify: All files formatted successfully
3. Read README.md and verify Training Outputs section is clear and accurate

---

## Dependencies & Execution Strategy

### User Story Dependencies

```
Setup (None)
    ↓
US1+US2 (P1) ← Independent MVP - can ship this alone
    ↓ (optional)
US3 (P2) ← Builds on US1+US2 checkpoint infrastructure

US5 (P1) ← Completely independent, can be done in parallel with US1+US2
```

### Parallel Execution Opportunities

**Phase 2 (US1+US2)**: All tasks sequential (same file modifications)

**Between Phases**:
- US3 and US5 can be implemented in parallel (different files)
- After US1+US2 complete, US3 and US5 are independent

**Phase 4 (US5)**: All tasks sequential (same file modifications)

**Phase 5 (Polish)**: T025 (format) must run before T026-T027 (documentation)

### MVP Strategy

**Minimum Viable Product**: US1+US2 only (T001-T012)
- Provides core value: dual logging for neural network training
- Enables monitoring and debugging during training
- Checkpoint improvements (US3) can be added later
- Genetic logging (US5) is independent add-on

**Recommended Delivery Order**:
1. Ship US1+US2 (neural network logging) - test and validate
2. Add US5 (genetic logging) in parallel or after US1+US2
3. Add US3 (enhanced checkpoints) when resume capability needed
4. Polish (format + docs) after all functionality complete

---

## Task Summary

**Total Tasks**: 27

**Breakdown by User Story**:
- Setup: 0 tasks
- US1+US2 (Neural Training Logging): 12 tasks
- US3 (Checkpoint Management): 4 tasks
- US5 (Genetic Logging): 8 tasks
- Polish: 3 tasks

**Parallel Opportunities**:
- T017 (US5 import) can start in parallel with US1+US2 if using separate working copies
- After US1+US2 complete: US3 and US5 can proceed in parallel

**Independent Test Criteria Met**:
- ✅ US1+US2: Run training, see console + log file
- ✅ US3: Load checkpoints, verify state
- ✅ US5: Run genetic, see console + genetic.log

**Format Validation**: ✅ All tasks follow checklist format with ID, Story label, and file paths
