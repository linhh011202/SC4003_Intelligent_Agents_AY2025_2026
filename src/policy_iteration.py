from __future__ import annotations

from .gridworld import Action, GridWorld, State
from .utils import initialize_utilities
from .value_iteration import convergence_threshold


def _initial_policy(env: GridWorld) -> dict[State, Action | None]:
    policy: dict[State, Action | None] = {}
    for state in env.get_states():
        actions = env.get_actions(state)
        policy[state] = actions[0] if actions else None
    return policy


def policy_iteration(
    env: GridWorld,
    epsilon: float = 1e-4,
    max_iterations: int = 1000,
    max_evaluation_sweeps: int = 10000,
) -> tuple[dict[State, Action | None], dict[State, float], list[dict[str, object]]]:
    utilities = initialize_utilities(env)
    policy = _initial_policy(env)
    threshold = convergence_threshold(env.gamma, epsilon)
    history: list[dict[str, object]] = []
    total_evaluation_sweeps = 0

    for iteration in range(1, max_iterations + 1):
        evaluation_delta = 0.0
        evaluation_sweeps = 0

        for _ in range(max_evaluation_sweeps):
            evaluation_sweeps += 1
            total_evaluation_sweeps += 1
            delta = 0.0
            next_utilities = dict(utilities)
            for state in env.get_non_terminal_states():
                action = policy[state]
                action_value = env.expected_action_value(state, action, utilities)
                delta = max(delta, abs(action_value - utilities[state]))
                next_utilities[state] = action_value
            utilities = next_utilities
            evaluation_delta = delta
            history.append(
                {
                    "iteration": float(total_evaluation_sweeps),
                    "outer_iteration": float(iteration),
                    "delta": delta,
                    "mean_utility": env.mean_utility(utilities),
                    "utilities": dict(utilities),
                }
            )
            if delta < threshold:
                break

        policy_changes = 0
        stable = True
        for state in env.get_non_terminal_states():
            old_action = policy[state]
            best_action = max(
                env.get_actions(state),
                key=lambda action: env.expected_action_value(state, action, utilities),
            )
            policy[state] = best_action
            if best_action != old_action:
                stable = False
                policy_changes += 1

        history[-1]["policy_changes"] = float(policy_changes)
        history[-1]["evaluation_sweeps"] = float(evaluation_sweeps)
        history[-1]["policy_stable"] = stable

        if stable:
            break

    return policy, utilities, history
