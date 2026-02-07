<!--
Sync Impact Report:
- Version change: 1.0.0 → 1.1.0
- Added principle: VI. User-Written AI Algorithms
- Principles defined: 6 (Learning Focus, Code Simplicity, No Packaging, Documentation Balance, Permission-First Operations, User-Written AI Algorithms)
- Templates requiring updates: N/A
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

**Rationale**: Consistency aids learning; automated formatting removes bikeshedding.

## Governance

This constitution guides development decisions. Amendments require:
1. Clear rationale for the change
2. User approval
3. Update to this document with version increment

All code contributions must align with the Core Principles. When principles conflict, Learning Focus (Principle I) takes precedence.

Violations should be caught in review, not after merge. If complexity is creeping in, simplify.

**Version**: 1.1.0 | **Ratified**: 2026-02-07 | **Last Amended**: 2026-02-07
