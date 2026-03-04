from __future__ import annotations

from typing import Dict, List, Tuple

from .gridworld import GridWorld, State
from .utils import initialize_utilities


def convergence_threshold(gamma: float, epsilon: float) -> float:
    if gamma <= 0.0:
        return epsilon
    if gamma >= 1.0:
        return epsilon
    return epsilon * (1.0 - gamma) / gamma


def value_iteration(
    env: GridWorld,
    epsilon: float = 1e-4,
    max_iterations: int = 10000,
) -> tuple[dict[State, float], list[dict[str, object]]]:
    utilities = initialize_utilities(env)
    threshold = convergence_threshold(env.gamma, epsilon)
    history: list[dict[str, object]] = []

    for iteration in range(1, max_iterations + 1):
        delta = 0.0
        next_utilities = dict(utilities)

        for state in env.get_non_terminal_states():
            best_value = max(
                env.expected_action_value(state, action, utilities)
                for action in env.get_actions(state)
            )
            delta = max(delta, abs(best_value - utilities[state]))
            next_utilities[state] = best_value

        utilities = next_utilities
        history.append(
            {
                "iteration": float(iteration),
                "delta": delta,
                "mean_utility": env.mean_utility(utilities),
                "utilities": dict(utilities),
            }
        )
        if delta < threshold:
            break

    return utilities, history
