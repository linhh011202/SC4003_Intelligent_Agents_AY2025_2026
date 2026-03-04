from __future__ import annotations

import random
from collections import deque


def _neighbors(state: tuple[int, int], size: int) -> list[tuple[int, int]]:
    row, col = state
    candidates = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
    return [
        candidate
        for candidate in candidates
        if 0 <= candidate[0] < size and 0 <= candidate[1] < size
    ]


def _has_path(
    size: int,
    walls: set[tuple[int, int]],
    start: tuple[int, int],
    targets: set[tuple[int, int]],
) -> bool:
    queue = deque([start])
    visited = {start}

    while queue:
        state = queue.popleft()
        if state in targets:
            return True
        for neighbor in _neighbors(state, size):
            if neighbor in visited or neighbor in walls:
                continue
            visited.add(neighbor)
            queue.append(neighbor)
    return False


def generate_maze_config(size: int, seed: int) -> dict:
    rng = random.Random(seed)
    start = (size - 1, 0)
    positive_anchor = (0, size - 1)

    reserved = {
        start,
        positive_anchor,
    }
    free_cells = [
        (row, col)
        for row in range(size)
        for col in range(size)
        if (row, col) not in reserved
    ]

    wall_count = min(len(free_cells) // 4, size + size // 2)
    negative_count = max(1, size // 3)
    positive_extra_count = max(1, size // 5)

    walls: set[tuple[int, int]] = set()
    candidates = free_cells[:]
    rng.shuffle(candidates)
    for cell in candidates:
        if len(walls) >= wall_count:
            break
        trial = set(walls)
        trial.add(cell)
        if _has_path(size, trial, start, {positive_anchor}):
            walls.add(cell)

    free_after_walls = [cell for cell in free_cells if cell not in walls]
    rng.shuffle(free_after_walls)

    reward_states: list[tuple[tuple[int, int], float]] = [(positive_anchor, 1.0)]
    used = {positive_anchor}

    for cell in free_after_walls:
        if len([item for item in reward_states if item[1] > 0]) >= positive_extra_count + 1:
            break
        if cell in used or cell == start:
            continue
        reward_states.append((cell, 1.0))
        used.add(cell)

    for cell in free_after_walls:
        if len([item for item in reward_states if item[1] < 0]) >= negative_count:
            break
        if cell in used or cell == start:
            continue
        reward_states.append((cell, -1.0))
        used.add(cell)

    rewards = [{"state": list(state), "value": value} for state, value in reward_states]
    walls_json = [list(state) for state in sorted(walls)]

    return {
        "rows": size,
        "cols": size,
        "walls": walls_json,
        "terminal_states": [],
        "rewards": rewards,
        "p_forward": 0.8,
        "gamma": 0.99,
        "step_reward": -0.05,
        "start_state": list(start),
    }
