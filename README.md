# Assignment 1: MDP Maze Solver

This project implements the core requirements for SC4003 Assignment 1 and is now managed with `uv`.

## Features

- GridWorld environment modeled as an MDP
- Value Iteration
- Policy Iteration
- PNG exports for:
  - original maze
  - optimal policy
  - utility heatmap
  - utility-estimate curves
  - Part 2 summary charts
- Deterministic maze generation for Part 2
- LaTeX report generated from actual run results
- `unittest` test suite
- Notebook-based visualization split into Part 1 and Part 2

## Project Layout

- `main.py`: CLI entrypoint
- `config/base.yaml`: base maze config stored as JSON-compatible YAML
- `src/gridworld.py`: MDP environment
- `src/value_iteration.py`: Value Iteration
- `src/policy_iteration.py`: Policy Iteration
- `src/maze_generator.py`: deterministic Part 2 maze generation
- `src/visualization.py`: PNG exports
- `src/reporting.py`: LaTeX report generation
- `src/notebook_helpers.py`: notebook display helpers
- `notebooks/part_1_visualization.ipynb`: Part 1 walkthrough
- `notebooks/part_2_visualization.ipynb`: Part 2 walkthrough
- `tests/test_assignment.py`: unit tests

## Assumption Used

The reward model follows the AIMA textbook (Eq. 17.5): `R(s)` is the immediate reward for being in state `s`, and the Bellman equation is `U(s) = R(s) + γ max_a Σ P(s'|s,a) U(s')`. There are no terminal states; green and brown cells are revisitable.

## Setup

Sync the project runtime:

```bash
uv sync
```

Install the notebook toolchain:

```bash
uv sync --group dev
```

## Run

Part 1:

```bash
uv run python main.py --part 1
```

Part 2:

```bash
uv run python main.py --part 2
```

Run everything:

```bash
uv run python main.py --part all
```

Run tests:

```bash
uv run python -m unittest discover -s tests -p 'test_*.py'
```

Open notebooks:

```bash
uv run --group dev jupyter notebook
```

## Output

Generated files are written under:

- `results/part1/`
- `results/part2/`
- `report/REPORT.tex`
- `report/REPORT.pdf`

Preferred visual files are the `.png` files in `results/`. Older `.svg` files may still exist from previous runs but are no longer used by the notebooks or report.
