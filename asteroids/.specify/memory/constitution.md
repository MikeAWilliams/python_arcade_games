<!--
Sync Impact Report:
- Version change: 1.2.0 → 1.3.0
- Modified sections: Code Quality Standards (added dual logging requirement)
- Added sections: None
- Removed sections: None
- Principles defined: 6 (unchanged)
- Templates requiring updates: N/A (plan-template Constitution Check is dynamic)
- Follow-up TODOs: None
-->

# Asteroids AI Learning Project Constitution

## Core Principles

### I. Learning Focus
This project exists for learning AI algorithms, not production deployment. Code decisions prioritize understanding over optimization. Experimentation is encouraged; premature abstraction is discouraged. Direct, readable implementations preferred over clever patterns.

**Rationale**: The primary value is educational. Code should teach, not impress.

### II. Code Simplicity
MUST keep code easily understood. MUST NOT introduce complex design patterns without explicit justification. Simple, direct solutions preferred over architectural complexity. Flat is better than nested.

**Rationale**: Complexity obscures learning. If the pattern needs explanation, it's probably too complex for this project.

### III. Source-Only Execution
MUST NOT package for distribution. All execution happens from source. No setup.py, no PyPI publishing, no pip installation complexities. Direct Python execution from project root.

**Rationale**: Packaging overhead distracts from AI learning goals. Source execution keeps it simple.

### IV. Documentation Balance
MUST keep documentation (README, CLAUDE.md) up to date with core functionality. Documentation MUST be brief and practical. Comments on key sections are valuable; excessive line-by-line commenting is distracting. MUST NOT generate extra documentation unless explicitly requested.

**Rationale**: Good docs enable learning; verbose docs hinder it. Focus on "how to use" over exhaustive API references.

### V. Permission-First Operations
MUST ask permission before performing file system operations (create, move, delete files/directories) and git operations (add, commit, push). User retains control over all persistent changes.

**Rationale**: Prevents unwanted changes to learning environment and maintains user agency over the codebase.

### VI. User-Written AI Algorithms
Agent MUST NOT write new core AI algorithms. The user writes AI implementations (heuristic AI logic, neural network architectures, training loops, reward functions). Agent MAY help debug existing AI code, explain concepts, suggest improvements, and assist with infrastructure/tooling.

**Rationale**: The learning value is in writing the algorithms. Agent writing them defeats the educational purpose. Debugging assistance preserves learning while reducing frustration.

## Project Structure

### Spec File Location
Specification files MUST be located in `/home/mike/source/python_arcade_games/asteroids/specs/` directory, NOT in the root repository folder. This project lives within a larger repository, and specs should be colocated with the project code.

**Rationale**: Keeps all project artifacts (code, specs, documentation) together in the asteroids subfolder, maintaining clear boundaries within the larger repository.

## Testing Policy

No unit tests required. Testing happens through:
- Running the game interactively
- Headless benchmarking with statistics
- Training runs with observable results

Empirical validation over test coverage.

**Rationale**: For a learning project, observable behavior is more valuable than test suites. Time spent writing tests could be spent learning AI.

## Code Quality Standards

- Use Black for formatting (enforced via `./format.sh`)
- Clear variable/function names over comments
- Type hints on function signatures
- Imports organized: stdlib → third-party → local
- Long-running scripts (training, data conversion, benchmarking) MUST use dual logging (console + file) instead of bare `print()`. The logging pattern is: `logging.getLogger()` with both a `StreamHandler(sys.stdout)` and a `FileHandler`, using `Formatter("%(message)s")`.

**Rationale**: Consistency aids learning; automated formatting removes bikeshedding. Dual logging preserves output from long-running operations that may take hours, allowing review after the fact without re-running.

## Governance

This constitution guides development decisions. Amendments require:
1. Clear rationale for the change
2. User approval
3. Update to this document with version increment

All code contributions must align with the Core Principles. When principles conflict, Learning Focus (Principle I) takes precedence.

Violations should be caught in review, not after merge. If complexity is creeping in, simplify.

**Version**: 1.3.0 | **Ratified**: 2026-02-07 | **Last Amended**: 2026-02-21
