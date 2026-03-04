from __future__ import annotations

import unittest

from src.gridworld import GridWorld
from src.maze_generator import generate_maze_config
from src.policy_iteration import policy_iteration
from src.utils import compare_policies, extract_policy
from src.value_iteration import value_iteration


class AssignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "rows": 3,
            "cols": 4,
            "walls": [[1, 1]],
            "terminal_states": [],
            "rewards": [
                {"state": [0, 3], "value": 1.0},
                {"state": [1, 3], "value": -1.0},
            ],
            "p_forward": 0.8,
            "gamma": 0.99,
            "step_reward": -0.05,
            "start_state": [2, 0],
        }
        self.env = GridWorld.from_config(self.config)

    def test_transition_probabilities_sum_to_one(self) -> None:
        probs = self.env.next_states_and_probs((2, 0), "UP")
        self.assertAlmostEqual(sum(prob for _, prob in probs), 1.0)

    def test_wall_and_boundary_keep_agent_in_place(self) -> None:
        self.assertEqual(self.env.move((0, 0), "UP"), (0, 0))
        self.assertEqual(self.env.move((1, 0), "RIGHT"), (1, 0))

    def test_value_and_policy_iteration_agree(self) -> None:
        vi_utilities, _ = value_iteration(self.env)
        vi_policy = extract_policy(self.env, vi_utilities)
        pi_policy, pi_utilities, _ = policy_iteration(self.env)

        max_diff = max(
            abs(vi_utilities[state] - pi_utilities[state]) for state in self.env.get_states()
        )
        self.assertLess(max_diff, 1e-3)
        self.assertEqual(compare_policies(self.env, vi_policy, pi_policy), 0)

    def test_reward_cells_are_not_terminal(self) -> None:
        self.assertFalse(self.env.is_terminal((0, 3)))
        self.assertEqual(sorted(self.env.get_actions((0, 3))), ["DOWN", "LEFT", "RIGHT", "UP"])

    def test_generated_maze_is_reproducible(self) -> None:
        first = generate_maze_config(size=9, seed=123)
        second = generate_maze_config(size=9, seed=123)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
