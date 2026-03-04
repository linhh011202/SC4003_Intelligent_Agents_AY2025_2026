from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict

from .gridworld import Action, GridWorld, State


ACTION_TO_GLYPH: dict[Action, str] = {
    "UP": "^",
    "RIGHT": ">",
    "DOWN": "v",
    "LEFT": "<",
}


def initialize_utilities(env: GridWorld) -> dict[State, float]:
    utilities: dict[State, float] = {}
    for state in env.get_states():
        utilities[state] = 0.0
    return utilities


def extract_policy(env: GridWorld, utilities: Dict[State, float]) -> dict[State, Action | None]:
    policy: dict[State, Action | None] = {}
    for state in env.get_states():
        if env.is_terminal(state):
            policy[state] = None
            continue
        best_action = None
        best_value = float("-inf")
        for action in env.get_actions(state):
            action_value = env.expected_action_value(state, action, utilities)
            if action_value > best_value:
                best_value = action_value
                best_action = action
        policy[state] = best_action
    return policy


def utilities_to_matrix(env: GridWorld, utilities: Dict[State, float]) -> list[list[str]]:
    matrix: list[list[str]] = []
    for row in range(env.rows):
        values: list[str] = []
        for col in range(env.cols):
            state = (row, col)
            if env.is_wall(state):
                values.append("WALL")
            else:
                values.append(f"{utilities[state]:.6f}")
        matrix.append(values)
    return matrix


def policy_to_grid(env: GridWorld, policy: dict[State, Action | None]) -> list[list[str]]:
    grid: list[list[str]] = []
    for row in range(env.rows):
        values: list[str] = []
        for col in range(env.cols):
            state = (row, col)
            if env.is_wall(state):
                values.append("WALL")
            else:
                action = policy[state]
                values.append(ACTION_TO_GLYPH[action] if action else ".")
        grid.append(values)
    return grid


def write_matrix_csv(path: Path, matrix: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(matrix)


def write_policy_txt(path: Path, env: GridWorld, policy: dict[State, Action | None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = policy_to_grid(env, policy)
    lines = ["\t".join(row) for row in grid]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_policies(
    env: GridWorld,
    left: dict[State, Action | None],
    right: dict[State, Action | None],
) -> int:
    mismatches = 0
    for state in env.get_non_terminal_states():
        if left[state] != right[state]:
            mismatches += 1
    return mismatches


def dump_metrics_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_state_utility_series(
    env: GridWorld,
    history: list[dict[str, object]],
    iteration_key: str = "iteration",
    snapshot_key: str = "utilities",
) -> dict[str, list[tuple[float, float]]]:
    series: dict[str, list[tuple[float, float]]] = {}
    for state in env.get_states():
        if env.is_wall(state):
            continue
        label = f"({state[0]},{state[1]})"
        points: list[tuple[float, float]] = []
        for entry in history:
            snapshot = entry[snapshot_key]
            points.append((float(entry[iteration_key]), float(snapshot[state])))
        series[label] = points
    return series
