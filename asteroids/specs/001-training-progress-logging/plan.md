# Implementation Plan: Training Progress Tracking and Logging

**Feature Branch**: `001-training-progress-logging`
**Created**: 2026-02-07
**Status**: Ready for Implementation

## Technical Context

### Current State Analysis

**Existing Code** (`training/main_pg.py`):
- Training loop prints progress to stdout every 500 epochs
- Model checkpoints saved to root directory: `model_epoch_{epoch}.pth`
- Final model saved to root directory: `nn_model.pth`
- No logging to files - all output goes only to screen
- Uses `print()` statements for all output
- Multi-process training with `ProcessPoolExecutor`

**Existing Code** (`training/main_genetic.py`):
- Genetic algorithm prints generation-by-generation progress
- Prints initial config, generation updates, and final results
- No logging to files - all output goes only to screen
- Uses `print()` statements (15+ locations)
- Multi-process evaluation with `ProcessPoolExecutor`

**Output Pattern**:
```
{epoch}/{total_epochs} -> avg_score:{avg_score:.2f}, max:{max_score:.2f}, loss:{loss:.4f} |
sim:{sim_time:.2f}s, train:{train_time:.2f}s |
elapsed:{elapsed_str}, total:{total_str}, remaining:{remaining_str}
```

### Technology Stack

- **Language**: Python 3.10+
- **ML Framework**: PyTorch
- **Logging**: Python stdlib `logging` module with custom formatter
- **File I/O**: Python stdlib `os`, `pathlib`
- **Datetime**: Python stdlib `datetime` for timestamps

### Dependencies

No new external dependencies required - using Python standard library only.

## Constitution Check

### Alignment with Core Principles

✅ **I. Learning Focus** - This is infrastructure/tooling work, not AI algorithm implementation. Aligns with "Agent MAY assist with infrastructure/tooling."

✅ **II. Code Simplicity** - Solution uses Python stdlib logging, no complex patterns. Direct file writes.

✅ **III. Source-Only Execution** - No packaging changes, runs from source.

✅ **IV. Documentation Balance** - Will update README with new checkpoint location and log file info. Keep brief.

✅ **V. Permission-First Operations** - Will ask before modifying `training/main_pg.py` and creating new files.

✅ **VI. User-Written AI Algorithms** - Not touching AI algorithms, only adding logging infrastructure.

### Quality Standards

- Use Black formatting (via `./format.sh`)
- Type hints on new functions
- Clear variable names
- Imports: stdlib → third-party (torch) → local (asteroids)

## Phase 0: Research & Design Decisions

### Decision 1: Logging Implementation Approach

**Chosen**: Python `logging` module with custom StreamHandler for dual output

**Rationale**:
- Standard library solution (no dependencies)
- Built-in support for multiple handlers (console + file simultaneously)
- Automatic timestamping and formatting
- Simple to add to existing `print()` statements

**Alternatives Considered**:
- Manual file writes alongside print: Requires duplicate code, error-prone
- `tee` command via subprocess: Platform-specific, adds complexity
- Third-party logging libraries (loguru): Adds dependency, violates simplicity

### Decision 2: Log File Naming Convention

**Chosen**: `training_YYYYMMDD_HHMMSS.log` in `nn_checkpoints/`

**Rationale**:
- ISO-8601-like format sorts chronologically
- Clear purpose prefix ("training_")
- No collision risk with concurrent runs
- Easy to identify by visual inspection

**Alternatives Considered**:
- Unix timestamp: Less human-readable
- UUID suffix: Unnecessarily complex
- Sequential numbers: Requires tracking state

### Decision 3: Checkpoint File Organization

**Chosen**: All checkpoints in `nn_checkpoints/`, prefix with timestamp

**Pattern**:
```
nn_checkpoints/
├── training_20260207_123045.log
├── checkpoint_20260207_123045_epoch_0.pth
├── checkpoint_20260207_123045_epoch_6000.pth
└── checkpoint_20260207_123045_final.pth
```

**Rationale**:
- Groups all artifacts from one training run
- Timestamp prefix makes cleanup easy (`rm checkpoint_20260207*`)
- Clear semantic names (best, final)

### Decision 4: Resume Training Implementation Scope

**Chosen**: P2 priority (checkpoints save state, but no auto-resume in this iteration)

**Rationale**:
- Spec marks this as P2
- Saving checkpoints enables future resume capability
- User can manually load checkpoint and modify script
- Auto-resume adds complexity (command-line flags, state management)
- Keeps implementation simple for learning project

**Implementation**: Save full state (model, optimizer, epoch, best_score) in checkpoint, but don't add resume logic yet.

## Phase 1: Implementation Design

### File Changes Required

**Modified Files**:
1. `training/main_pg.py` - Add logging setup, update checkpoint paths
2. `training/main_genetic.py` - Add logging setup to root directory

**No New Files** - Using existing training scripts only

### Core Components

#### Component 1: Logging Setup

**Location**: `training/main_pg.py` (top of `train_model()` function)

**Functionality**:
```python
def setup_logging(log_dir="nn_checkpoints"):
    """
    Set up dual logging (console + file).
    Returns: logger instance, timestamp for filenames
    """
    # Create directory if needed
    os.makedirs(log_dir, exist_ok=True)

    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")

    # Configure logger
    logger = logging.getLogger("training")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)

    # Simple format (no logger name prefix)
    formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger, timestamp
```

#### Component 2: Checkpoint Saving

**Location**: `training/main_pg.py` (replace existing save logic)

**Changes**:
```python
# At epoch intervals
checkpoint_path = os.path.join(
    "nn_checkpoints",
    f"checkpoint_{timestamp}_epoch_{epoch}.pth"
)
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': opt.state_dict(),
    'max_score': max_score,
    'loss': loss,
}, checkpoint_path)

# Final model (at end of training)
final_checkpoint_path = os.path.join(
    "nn_checkpoints",
    f"checkpoint_{timestamp}_final.pth"
)
torch.save({
    'epoch': total_epochs - 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': opt.state_dict(),
    'max_score': max_score,
    'loss': loss,
}, final_checkpoint_path)
```

#### Component 3: Print Statement Migration

**Location**: Throughout `training/main_pg.py`

**Pattern**:
```python
# Before
print(f"Using device: {device}")

# After
logger.info(f"Using device: {device}")
```

**All print statements** to migrate:
1. Line 196: Device info
2. Line 201: Worker count
3. Line 202: Batch size
4. Line 215: Worker pool creation
5. Line 256-260: Epoch progress
6. Line 266: Training completion time
7. Line 267: Model saved message

#### Component 4: Genetic Algorithm Logging Setup

**Location**: `training/main_genetic.py` (before `main()` function)

**Functionality**:
```python
def setup_genetic_logging():
    """
    Set up dual logging (console + file) for genetic algorithm.
    Log file saved to root directory as genetic.log (overwritten each run).
    Returns: logger instance
    """
    log_file = "genetic.log"

    # Configure logger
    logger = logging.getLogger("genetic")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # File handler (overwrite mode)
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)

    # Simple format (no logger name prefix)
    formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
```

#### Component 5: Print Statement Migration (main_genetic.py)

**Location**: Throughout `training/main_genetic.py`

**Pattern**: Same as main_pg.py (replace `print()` with `logger.info()`)

**Print statements to migrate** (~15 locations):
- Lines 237-240: Initial algorithm configuration
- Line 315: Generation progress updates
- Lines 352-376: Final results and best parameters display

**Note**: Pass logger instance through method calls or make it a class attribute for clean access.

### Implementation Steps

#### For training/main_pg.py:

1. **Add imports** at top of file:
   ```python
   import logging
   from datetime import datetime
   from pathlib import Path
   ```

2. **Add `setup_logging()` function** before `train_model()`

3. **In `train_model()` function**:
   - Call `setup_logging()` at start, get logger and timestamp
   - Replace all `print()` calls with `logger.info()`
   - Update checkpoint save paths to use `nn_checkpoints/` and timestamp
   - Update final model save to use new checkpoint format

#### For training/main_genetic.py:

4. **Add import** at top of file:
   ```python
   import logging
   ```

5. **Add `setup_genetic_logging()` function** before `main()`

6. **In `run()` method of GeneticAlgorithm class**:
   - Call `setup_genetic_logging()` at start of method, get logger
   - Store logger as instance variable: `self.logger = setup_genetic_logging()`
   - Replace all `print()` calls with `self.logger.info()`

7. **In `main()` function**:
   - Pass logger or call `setup_genetic_logging()` at start
   - Replace final `print()` calls with `logger.info()`

#### Testing:

8. **Format code**: Run `./format.sh`

9. **Test main_pg.py**:
   - Run short training (small epoch count)
   - Verify log file created in `nn_checkpoints/`
   - Verify checkpoints saved with correct naming
   - Verify console output still appears
   - Check log file contains same content as console

10. **Test main_genetic.py**:
    - Run short genetic run (few generations)
    - Verify `genetic.log` created in root directory
    - Verify console output still appears
    - Check log file contains same content as console
    - Run again and verify genetic.log is overwritten

### Edge Case Handling

1. **Missing `nn_checkpoints/` directory**:
   - Handled by `os.makedirs(log_dir, exist_ok=True)` in `setup_logging()`

2. **Disk space exhaustion**:
   - PyTorch `torch.save()` will raise `OSError` - let it bubble up with clear error
   - No special handling needed (fail fast principle)

3. **Log file name collision** (same second):
   - Very unlikely with HHMMSS precision
   - If it happens, file would be overwritten (mode='w')
   - Acceptable for learning project (not production)

4. **Logging failure during training**:
   - If file handler fails, console handler continues
   - If logger completely fails, Python will raise exception
   - No silent failures - explicit errors preferred for learning

## Testing Plan

### Manual Test Cases

**Test 1: Verify Dual Logging**
```bash
python training/main_pg.py --batch-size 2 --workers 2
# Let run for ~10 epochs, then Ctrl+C
# Check: console shows output
# Check: log file exists in nn_checkpoints/
# Check: log file content matches what was on console
```

**Test 2: Verify Checkpoint Saving**
```bash
python training/main_pg.py --batch-size 2 --workers 2
# Let run through at least one checkpoint save
# Check: checkpoint file created in nn_checkpoints/
# Check: filename includes timestamp
# Check: can load checkpoint with torch.load()
```

**Test 3: Verify Directory Creation**
```bash
rm -rf nn_checkpoints/  # Remove directory
python training/main_pg.py --batch-size 2 --workers 2
# Check: training starts without error
# Check: nn_checkpoints/ directory created automatically
```

**Test 5: Verify Genetic Algorithm Logging**
```bash
python training/main_genetic.py --population 10 --generations 3
# Let run to completion
# Check: console shows generation progress
# Check: genetic.log created in root directory
# Check: genetic.log contains all console output (headers, generation stats, final results)
```

**Test 6: Verify Genetic Log Overwrite**
```bash
python training/main_genetic.py --population 10 --generations 2
python training/main_genetic.py --population 10 --generations 2  # Run again
# Check: genetic.log contains only second run's output
# Check: first run's data is not present
```

### Success Criteria Verification

- **SC-001** (Real-time monitoring): ✓ Console output still shows progress
- **SC-002** (100% capture): ✓ Log file contains all console output
- **SC-003** (Resume capability): ✓ Checkpoints save full state (manual resume possible)
- **SC-004** (Timestamp distinction): ✓ Filenames include YYYYMMDD_HHMMSS
- **SC-005** (Artifacts in nn_checkpoints): ✓ All files go to nn_checkpoints/

## Documentation Updates

### README.md Changes

Add to "Training" section:

```markdown
### Training Outputs

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
```

### No CLAUDE.md Changes

The `CLAUDE.md` already documents the training scripts. No updates needed since the interface (`python training/main_pg.py`) remains the same.

## Risk Assessment

### Low Risk Items

✅ Adding logging - Non-invasive, doesn't change training logic
✅ Changing save paths - Simple string modification
✅ Directory creation - Standard library, well-tested

### Medium Risk Items

⚠️ **Replacing all print statements**: Easy to miss one, but testable
- **Mitigation**: Search for all `print(` in file before committing

⚠️ **Checkpoint format change**: Old code expects `nn_model.pth` in root
- **Mitigation**: Keep backward compatibility - save final model to both locations temporarily

### No High Risk Items

## Implementation Checklist

### main_pg.py:
- [ ] Add imports (logging, datetime, pathlib)
- [ ] Create `setup_logging()` function
- [ ] Call `setup_logging()` in `train_model()`
- [ ] Replace all print() with logger.info()
- [ ] Update checkpoint save paths
- [ ] Update final model save

### main_genetic.py:
- [ ] Add import (logging)
- [ ] Create `setup_genetic_logging()` function
- [ ] Call `setup_genetic_logging()` in `run()` method
- [ ] Store logger as instance variable
- [ ] Replace all print() with self.logger.info() (in run() method)
- [ ] Replace print() in main() function

### Testing & Documentation:
- [ ] Run `./format.sh`
- [ ] Test main_pg.py dual logging
- [ ] Test main_pg.py checkpoint saving
- [ ] Test main_pg.py directory creation
- [ ] Test main_genetic.py dual logging
- [ ] Test main_genetic.py log overwrite
- [ ] Update README.md
- [ ] Verify all success criteria met

## Future Enhancements (Out of Scope)

These are explicitly deferred for simplicity:

1. **Auto-resume training**: Add `--resume` flag to load checkpoint
2. **Checkpoint cleanup**: Auto-delete old checkpoints
3. **Structured logging**: JSON format for machine parsing
4. **Progress bars**: Visual progress indicators (tqdm)
5. **Real-time plotting**: Matplotlib integration
6. **Metrics dashboard**: Web-based monitoring

## Next Steps

Ready to implement. Agent should:

1. Ask permission to modify `training/main_pg.py`
2. Make all changes in one edit session
3. Run `./format.sh`
4. Ask user to test with a short training run
5. Update README.md after successful test
6. Await user approval before git commit
