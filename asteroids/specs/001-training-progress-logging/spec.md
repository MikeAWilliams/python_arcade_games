# Feature Specification: Training Progress Tracking and Logging

**Feature Branch**: `001-training-progress-logging`
**Created**: 2026-02-07
**Status**: Draft
**Input**: User description: "update the existing source to better track results and progress during long runs. checkpoint files and final weights should be written to nn_checkpoints. All std_out should go both to screen and to a log file in nn_checkpoints."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Monitor Training Progress in Real-Time (Priority: P1)

As a user running neural network training, I need to see training progress on screen as it happens so I can verify the training is proceeding correctly and catch issues early.

**Why this priority**: Core capability - without real-time feedback, users have no visibility into whether training is working.

**Independent Test**: Run training script and observe epoch updates, loss values, and score statistics appearing on screen during execution.

**Acceptance Scenarios**:

1. **Given** training has started, **When** an epoch completes, **Then** epoch number, loss, and performance metrics are displayed on screen
2. **Given** training is running, **When** I watch the screen output, **Then** I can see progress updates without needing to check files

---

### User Story 2 - Review Training History After Completion (Priority: P1)

As a user who has completed a training run, I need to review the complete training history from log files so I can analyze performance trends and debug issues that occurred during training.

**Why this priority**: Essential for learning and debugging - logs preserve the full history that may scroll off screen.

**Independent Test**: Complete a training run, then open the log file in `nn_checkpoints/` and verify it contains all output that appeared on screen.

**Acceptance Scenarios**:

1. **Given** training has completed, **When** I open the log file in `nn_checkpoints/`, **Then** I see all console output including epoch stats, errors, and progress messages
2. **Given** a training run encountered an error, **When** I review the log file, **Then** I can see the full error message and context leading up to it
3. **Given** multiple training runs, **When** I check `nn_checkpoints/`, **Then** each run has its own timestamped log file

---

### User Story 3 - Resume Training from Checkpoints (Priority: P2)

As a user whose training was interrupted, I need checkpoint files saved in `nn_checkpoints/` so I can resume training from the last saved state instead of starting over.

**Why this priority**: Important for long runs and experimentation, but not required for basic training functionality.

**Independent Test**: Start training, stop it mid-run, verify checkpoint file exists in `nn_checkpoints/`, restart training and verify it resumes from saved state.

**Acceptance Scenarios**:

1. **Given** training is running, **When** each epoch completes, **Then** a checkpoint file is written to `nn_checkpoints/`
2. **Given** training was stopped early, **When** I restart with the same parameters, **Then** training resumes from the last checkpoint
3. **Given** training completes successfully, **When** I check `nn_checkpoints/`, **Then** the final model weights are saved there

---

---

### User Story 5 - Monitor Genetic Algorithm Progress (Priority: P1)

As a user running genetic algorithm optimization, I need to see generation-by-generation progress on screen and have it logged to a file so I can track evolution and review results later.

**Why this priority**: Essential for understanding genetic algorithm convergence and debugging parameter evolution.

**Independent Test**: Run genetic training, observe generation updates on screen, verify `genetic.log` in root directory contains all output.

**Acceptance Scenarios**:

1. **Given** genetic training has started, **When** each generation completes, **Then** generation number, best fitness, and population statistics are displayed on screen
2. **Given** genetic training is running, **When** I check the root directory, **Then** `genetic.log` file exists and contains all console output
3. **Given** I start a new genetic training run, **When** `genetic.log` already exists, **Then** it is overwritten with the new run's output

---

### Edge Cases

- When `nn_checkpoints/` directory doesn't exist run should terminate with an error message.
- How does the system handle disk space exhaustion during checkpoint writes? (Training should fail gracefully with clear error message)
- If a log file already exists clear and overwrite it
- How does logging behave if stdout fails? (Training should continue but log the logging failure)

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST write all console output to both screen (stdout) and a log file simultaneously
- **FR-002**: Neural network training (main_pg.py) MUST save log files to the `nn_checkpoints/` directory
- **FR-003**: System MUST create `nn_checkpoints/` directory automatically if it doesn't exist
- **FR-004**: System MUST save model checkpoint files to `nn_checkpoints/` directory
- **FR-005**: System MUST save final trained model weights to `nn_checkpoints/` directory
- **FR-006**: Neural network training log files MUST include timestamps for each training run to prevent overwrites
- **FR-007**: Checkpoint files MUST include sufficient information to resume training (model weights, optimizer state, epoch number)
- **FR-008**: System MUST display epoch number, loss values, and performance metrics during training
- **FR-009**: Neural network training log file naming MUST include date/time to distinguish between multiple training runs
- **FR-010**: Genetic algorithm training (main_genetic.py) MUST save log file as `genetic.log` in the project root directory
- **FR-011**: Genetic algorithm log file MUST be overwritten on each new run (no timestamp needed)

### Key Entities

- **Training Run**: A single execution of the training script, characterized by start time, parameters, and output log
- **Checkpoint**: A snapshot of training state at a specific epoch, including model weights and training metadata
- **Log File**: Text file containing all console output from a training run, stored in `nn_checkpoints/` with timestamp in filename

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: User can monitor training progress in real-time without checking files
- **SC-002**: 100% of console output is captured in log files for post-run analysis
- **SC-003**: Training runs can be resumed from checkpoints without losing progress
- **SC-004**: Log files from different training runs can be distinguished by timestamp in filename
- **SC-005**: User can locate all training artifacts (logs, checkpoints, final weights) in the `nn_checkpoints/` directory

## Assumptions

- Training scripts are in the `training/` directory (`training/main_pg.py` and `training/main_genetic.py`)
- Current training output goes to stdout (standard Python print statements)
- Checkpoint saving frequency will be per-epoch (reasonable default for iterative training)
- Log file format will be plain text (simplest for grep/analysis)
- Timestamp format for neural network log files will be ISO-8601 compatible for sorting (e.g., `YYYYMMDD_HHMMSS`)
- Genetic algorithm runs infrequently enough that a single `genetic.log` file (overwritten each run) is sufficient

## Scope

### In Scope

- Dual logging (screen + file) for both training scripts
- Neural network training: Checkpoint file management in `nn_checkpoints/`
- Neural network training: Final model weight storage in `nn_checkpoints/`
- Neural network training: Timestamped log file naming
- Genetic algorithm training: Simple `genetic.log` in root directory (overwritten each run)

### Out of Scope

- Real-time plotting/graphing of metrics
- Web dashboard for monitoring
- Distributed training support
- Checkpoint compression or optimization
- Automatic cleanup of old checkpoints
- Email/notification alerts for training completion
- Integration with MLOps platforms (MLflow, Weights & Biases, etc.)
