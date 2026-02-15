# Feature Specification: Cross-Entropy Supervised Learning Training Script

**Feature Branch**: `002-cross-entropy-training`
**Created**: 2026-02-07
**Status**: Draft
**Input**: User description: "Add a new script training/cross_entropy.py which will implement supervised learning using cross entropy. Start by adding a file with the required basic main and argument parsing. It should log similar to the other training scripts and use the nn_checkpoints folder as output. It should accept a base_name argument used to name its files. Also use the base_name to load data files with that base name from the data folder. After the base script, with main only is defined, the user will implement the majority of the code manually, and will define tasks ad hock as needed."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Create Basic Training Script Structure (Priority: P1)

As a developer, I need a basic cross-entropy training script with standard argument parsing and logging infrastructure so I can implement the supervised learning algorithm without worrying about boilerplate setup.

**Why this priority**: This is the foundation - without the script structure, logging, and argument parsing in place, no training can occur. This is the MVP scaffolding.

**Independent Test**: Run `python training/cross_entropy.py --help` and verify it shows all expected arguments. Run with `--base-name test_run` and verify log file is created in `nn_checkpoints/` with timestamp.

**Acceptance Scenarios**:

1. **Given** the script exists, **When** I run with `--help`, **Then** all command-line arguments are documented
2. **Given** I run the script with `--base-name my_data`, **When** execution starts, **Then** a timestamped log file is created in `nn_checkpoints/` directory
3. **Given** the script is running, **When** output is generated, **Then** it appears on both console and in the log file simultaneously

---

### User Story 2 - Load Training Data from Files (Priority: P2)

As a developer implementing supervised learning, I need the script to load training data from the data directory using the provided base name so I can train models on recorded game data.

**Why this priority**: Data loading is essential for training but is independent of the core training algorithm. Can be tested by verifying files are found and loaded correctly.

**Independent Test**: Place test `.npz` files in `data/` with pattern `<basename>_*.npz`, run script with matching `--base-name`, verify script reports finding and attempting to load the files.

**Acceptance Scenarios**:

1. **Given** data files exist with pattern `data/<basename>_*.npz`, **When** script runs with matching `--base-name`, **Then** all matching files are discovered
2. **Given** data files are found, **When** script attempts to load them, **Then** errors are logged clearly if files are corrupt or missing expected data
3. **Given** no data files match the base name, **When** script runs, **Then** it reports an error and exits gracefully

---

### User Story 3 - Save Training Outputs to nn_checkpoints (Priority: P2)

As a developer training models, I need all training outputs (checkpoints, logs, final models) saved to the `nn_checkpoints/` directory with consistent naming based on the base name so I can organize and track different training runs.

**Why this priority**: Output organization is important for reproducibility but doesn't affect the core training algorithm. Can be implemented and tested independently.

**Independent Test**: Run training script and verify all output files are created in `nn_checkpoints/` with filenames incorporating the base name and timestamps.

**Acceptance Scenarios**:

1. **Given** training is in progress, **When** a checkpoint is saved, **Then** it goes to `nn_checkpoints/` with format `<basename>_YYYYMMDD_HHMMSS_*.pth`
2. **Given** training completes, **When** final model is saved, **Then** it goes to `nn_checkpoints/` with a clear final model indicator
3. **Given** the `nn_checkpoints/` directory doesn't exist, **When** script starts, **Then** the directory is created automatically

---

### Edge Cases

- What happens when `data/` directory doesn't exist? (Script should create it or report clear error)
- How does the system handle when no data files match the base name? (Should error with helpful message)
- What happens if `nn_checkpoints/` can't be created due to permissions? (Should fail with clear error message)
- How does logging behave if disk is full during training? (Should fail gracefully with error logged)

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Script MUST accept a `--base-name` command-line argument to identify which data files to load and how to name output files
- **FR-002**: Script MUST load training data from `data/` directory using pattern `data/<base_name>_*.npz`
- **FR-003**: Script MUST save all output files (logs, checkpoints, models) to `nn_checkpoints/` directory
- **FR-004**: Script MUST create timestamped log files in `nn_checkpoints/` with format `<base_name>_YYYYMMDD_HHMMSS.log`
- **FR-005**: Script MUST write all console output to both stdout and the log file simultaneously (dual logging)
- **FR-006**: Script MUST create `nn_checkpoints/` directory automatically if it doesn't exist
- **FR-007**: Script MUST provide clear error messages when data files are not found or cannot be loaded
- **FR-008**: Script MUST include standard training arguments (batch size, learning rate, epochs, etc.) with reasonable defaults
- **FR-009**: Script MUST save model checkpoints with filenames based on base name and timestamp to prevent overwrites
- **FR-010**: Script MUST follow the same logging and output patterns as existing training scripts (`policy_gradient.py`, `genetic.py`)

### Key Entities

- **Training Script**: The main executable that orchestrates supervised learning with cross-entropy loss
- **Training Data**: NPZ files containing recorded game states and actions from the data directory
- **Training Run**: A single execution of the script, identified by base name and timestamp
- **Checkpoint**: A saved model state at a specific point during training, stored in `nn_checkpoints/`
- **Log File**: Text file containing all console output from a training run, stored in `nn_checkpoints/`

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Developer can run the script with `--help` and see all available arguments documented
- **SC-002**: Running the script creates a timestamped log file in `nn_checkpoints/` containing all console output
- **SC-003**: Script automatically discovers all data files matching the provided base name pattern
- **SC-004**: All training artifacts (logs, checkpoints) are organized in `nn_checkpoints/` with consistent naming
- **SC-005**: Script provides clear error messages when data files are missing or invalid, allowing quick debugging
- **SC-006**: Training outputs follow the same organizational pattern as other training scripts in the project

## Assumptions

- Training data files are in NPZ format (NumPy compressed archive)
- Data files follow the naming convention `<base_name>_<suffix>.npz` in the `data/` directory
- The user will implement the core training algorithm manually after the scaffolding is in place
- Logging format and dual output behavior should match `policy_gradient.py` and `genetic.py`
- Timestamp format for log files will be `YYYYMMDD_HHMMSS` for consistency
- Standard Python logging module will be used (no external logging frameworks)
- The script will be run from the project root directory

## Scope

### In Scope

- Basic script structure with main() function and argument parsing
- Dual logging setup (console + file in `nn_checkpoints/`)
- Command-line argument handling for base name, batch size, learning rate, epochs
- Data file discovery and loading infrastructure
- Output file organization in `nn_checkpoints/`
- Error handling for common failure cases (missing files, directory creation)
- Timestamped file naming to prevent overwrites

### Out of Scope

- Core supervised learning algorithm implementation (user will implement manually)
- Neural network architecture definition (user's responsibility)
- Cross-entropy loss function implementation (user will implement)
- Training loop logic (user will implement)
- Data preprocessing and augmentation
- Hyperparameter tuning or optimization
- Model evaluation and validation logic
- Integration with experiment tracking platforms
- Distributed training support
- GPU acceleration (though arguments for device selection can be included)
