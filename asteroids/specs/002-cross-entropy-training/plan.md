# Implementation Plan: Cross-Entropy Training Script Scaffolding

**Feature Branch**: `002-cross-entropy-training`
**Created**: 2026-02-07
**Status**: Ready for Implementation

## Overview

This is a **scaffolding feature** - the agent creates the basic script structure (argument parsing, logging, file I/O), and the user implements the core supervised learning algorithm manually.

**Agent Responsibility**: Infrastructure and boilerplate
**User Responsibility**: Cross-entropy loss, training loop, neural network architecture

## Technical Context

### Current State Analysis

**Existing Training Scripts**:
- `training/policy_gradient.py` - Uses REINFORCE policy gradient with dual logging to `nn_checkpoints/`
- `training/genetic.py` - Genetic algorithm with dual logging to `genetic.log` in root

**Common Patterns Observed**:
```python
# Logging setup
def setup_logging(log_dir="nn_checkpoints"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    logger = logging.getLogger("training")
    # ... dual handler setup (console + file)
    return logger, timestamp

# Argument parsing
parser = argparse.ArgumentParser(description="...")
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--workers", type=int, default=None)
# ... etc

# Directory creation
os.makedirs("nn_checkpoints", exist_ok=True)
```

**New Requirements for cross_entropy.py**:
- Accept `--base-name` argument (new pattern, not used by other scripts)
- Use base name for both input (loading data files) and output (naming log/checkpoint files)
- Load data from `data/<basename>_*.npz` files
- Create log file as `nn_checkpoints/<basename>_YYYYMMDD_HHMMSS.log`
- Save checkpoints as `nn_checkpoints/<basename>_YYYYMMDD_HHMMSS_*.pth`

### Technology Stack

- **Language**: Python 3.10+
- **Argument Parsing**: argparse (stdlib)
- **Logging**: logging module (stdlib) with dual handlers
- **File Discovery**: glob.glob() for pattern matching
- **Data Loading**: numpy.load() for NPZ files
- **Model Saving**: torch.save() for checkpoints
- **Datetime**: datetime.now().strftime() for timestamps

### Dependencies

All stdlib except:
- numpy (already in requirements.txt) - for loading NPZ files
- torch (already in requirements.txt) - for saving model checkpoints

No new dependencies required.

## Constitution Check

### Alignment with Core Principles

✅ **I. Learning Focus** - Agent creates infrastructure only. User writes the learning algorithm.

✅ **II. Code Simplicity** - Using same patterns as existing scripts. No complex abstractions.

✅ **III. Source-Only Execution** - Script runs directly from `training/` directory.

✅ **IV. Documentation Balance** - Will update README with new script usage. Keep brief.

✅ **V. Permission-First Operations** - Will ask before creating `training/cross_entropy.py` file.

✅ **VI. User-Written AI Algorithms** - Agent does NOT implement:
  - Cross-entropy loss function
  - Training loop logic
  - Neural network architecture
  - Data preprocessing
  - Model evaluation

  Agent ONLY implements:
  - Argument parsing
  - Logging setup
  - File I/O (discover, load, save)
  - Main() entry point with TODO comments for user

### Quality Standards

- Use Black formatting (via `./format.sh`)
- Type hints on function signatures
- Clear variable names
- Imports: stdlib → third-party (numpy, torch) → local (asteroids.*)

## Implementation Design

### File to Create

**training/cross_entropy.py** (NEW FILE)

Structure:
```python
"""
Cross-entropy supervised learning for Asteroids AI.

User implements: Training loop, loss function, model architecture
Agent provides: Argument parsing, logging, file I/O scaffolding
"""

import argparse
import glob
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Add parent to path for asteroids imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def setup_logging(base_name):
    """
    Set up dual logging (console + file).

    Args:
        base_name: Base name for log file

    Returns:
        logger instance, timestamp string
    """
    # Create nn_checkpoints directory
    os.makedirs("nn_checkpoints", exist_ok=True)

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join("nn_checkpoints", f"{base_name}_{timestamp}.log")

    # Configure logger (same pattern as policy_gradient.py)
    logger = logging.getLogger("cross_entropy")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)

    # Simple format
    formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger, timestamp


def load_training_data(base_name, logger):
    """
    Load training data from data/<base_name>_*.npz files.

    Args:
        base_name: Base name pattern for data files
        logger: Logger instance

    Returns:
        List of loaded numpy archives (or processed data)

    Raises:
        FileNotFoundError: If no matching files found
    """
    # Find all matching files
    pattern = f"data/{base_name}_*.npz"
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No data files found matching pattern: {pattern}\n"
            f"Please ensure data files exist in data/ directory."
        )

    logger.info(f"Found {len(files)} data files matching '{base_name}'")

    # Load files
    loaded_data = []
    for filepath in files:
        try:
            data = np.load(filepath)
            loaded_data.append(data)
            logger.info(f"  Loaded: {filepath}")
        except Exception as e:
            logger.error(f"  Failed to load {filepath}: {e}")
            raise

    logger.info(f"Successfully loaded {len(loaded_data)} files")

    # TODO: User implements data processing/concatenation
    return loaded_data


def train_model(base_name, batch_size, learning_rate, epochs, device):
    """
    Train model using cross-entropy loss on supervised data.

    Args:
        base_name: Base name for data files and output files
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        epochs: Number of training epochs
        device: Device to train on ('cpu' or 'cuda')
    """
    # Set up logging
    logger, timestamp = setup_logging(base_name)

    logger.info(f"Cross-Entropy Supervised Learning")
    logger.info(f"Base name: {base_name}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Device: {device}")
    logger.info("")

    # Load training data
    logger.info("Loading training data...")
    data = load_training_data(base_name, logger)

    # TODO: User implements
    # - Data preprocessing and batching
    # - Model architecture definition
    # - Loss function (cross-entropy)
    # - Optimizer setup
    # - Training loop
    # - Checkpoint saving

    logger.info("TODO: Implement training loop")
    logger.info("  1. Preprocess data into batches")
    logger.info("  2. Define neural network model")
    logger.info("  3. Define cross-entropy loss function")
    logger.info("  4. Create optimizer (e.g., Adam)")
    logger.info("  5. Implement training loop:")
    logger.info("     - Forward pass")
    logger.info("     - Compute loss")
    logger.info("     - Backward pass")
    logger.info("     - Optimizer step")
    logger.info("  6. Save checkpoints periodically")
    logger.info(f"  7. Save final model to nn_checkpoints/{base_name}_{timestamp}_final.pth")

    # Placeholder return
    logger.info("")
    logger.info("Training scaffolding complete. User should implement algorithm.")


def main():
    parser = argparse.ArgumentParser(
        description="Train neural network using supervised learning with cross-entropy loss"
    )

    # Required arguments
    parser.add_argument(
        "--base-name",
        type=str,
        required=True,
        help="Base name for data files (data/<base_name>_*.npz) and output files"
    )

    # Training hyperparameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )

    # Device selection
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to train on (default: cpu)"
    )

    args = parser.parse_args()

    # Run training
    train_model(
        base_name=args.base_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        device=args.device
    )


if __name__ == "__main__":
    main()
```

### Implementation Steps

1. **Create training/cross_entropy.py** with scaffolding code above
2. **Run ./format.sh** to format the new file
3. **Test basic functionality**:
   ```bash
   python training/cross_entropy.py --help
   # Should show all arguments

   python training/cross_entropy.py --base-name test_data
   # Should error with "No data files found" (expected - no test data exists)
   ```
4. **Update README.md** with new script documentation
5. **Commit the scaffolding**

### README.md Updates

Add to Training section:

```markdown
**Train with Supervised Learning (Cross-Entropy):**
```bash
python training/cross_entropy.py --base-name heuristic_data --batch-size 64 --epochs 200
```

#### Training Outputs

**Cross-Entropy Training** (`training/cross_entropy.py`):

All artifacts saved to `nn_checkpoints/`:

- **Log Files**: `<basename>_YYYYMMDD_HHMMSS.log`
  - Complete console output from training run
  - Includes data loading, training progress, errors

- **Checkpoints**: `<basename>_YYYYMMDD_HHMMSS_*.pth`
  - Saved during training (user implements checkpoint logic)
  - Final model saved as `<basename>_YYYYMMDD_HHMMSS_final.pth`

**Note**: Core training algorithm not implemented - user implements cross-entropy loss and training loop.
```

## Testing Plan

### Manual Tests

**Test 1: Verify Help Text**
```bash
python training/cross_entropy.py --help
# Check: All arguments documented
# Check: --base-name is required
```

**Test 2: Verify Logging Setup**
```bash
python training/cross_entropy.py --base-name test_run
# Should error with "No data files found"
# BUT should create log file first
# Check: nn_checkpoints/test_run_YYYYMMDD_HHMMSS.log exists
# Check: Log contains error message
```

**Test 3: Verify Directory Creation**
```bash
rm -rf nn_checkpoints/
python training/cross_entropy.py --base-name test_run
# Check: nn_checkpoints/ created automatically
```

## Risk Assessment

### Low Risk

✅ Creating new file (doesn't modify existing code)
✅ Using existing patterns from other training scripts
✅ All TODO comments make user responsibilities clear

### No Medium/High Risk Items

This is scaffolding only. No complex logic to implement.

## Implementation Checklist

- [ ] Create training/cross_entropy.py with scaffolding code
- [ ] Run ./format.sh to format code
- [ ] Test --help output
- [ ] Test logging setup (even with missing data)
- [ ] Update README.md Training section
- [ ] Commit scaffolding

## Future Work (User Implements)

After scaffolding is in place, user will implement:

1. Data preprocessing logic
2. Neural network model architecture
3. Cross-entropy loss function
4. Training loop with forward/backward passes
5. Checkpoint saving during training
6. Model evaluation/validation logic

User will create ad-hoc tasks as needed for these implementations.

## Next Steps

Ready to implement. Agent should:

1. Ask permission to create `training/cross_entropy.py`
2. Create file with scaffolding code above
3. Run `./format.sh`
4. Test basic functionality (--help, logging)
5. Update README.md
6. Await user approval before git commit
