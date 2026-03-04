from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


State = Tuple[int, int]
Action = str


ACTION_TO_DELTA: dict[Action, tuple[int, int]] = {
    "UP": (-1, 0),
    "RIGHT": (0, 1),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
}

LEFT_OF: dict[Action, Action] = {
    "UP": "LEFT",
    "RIGHT": "UP",
    "DOWN": "RIGHT",
    "LEFT": "DOWN",
}

RIGHT_OF: dict[Action, Action] = {
    "UP": "RIGHT",
    "RIGHT": "DOWN",
    "DOWN": "LEFT",
    "LEFT": "UP",
}


@dataclass(frozen=True)
class GridWorld:
    rows: int
    cols: int
    walls: frozenset[State]
    rewards: Dict[State, float]
    terminal_states: frozenset[State]
    p_forward: float
    gamma: float
    step_reward: float
    start_state: State | None = None

    @classmethod
    def from_config(cls, config: dict) -> "GridWorld":
        walls = frozenset(tuple(item) for item in config["walls"])
        rewards = {
            tuple(item["state"]): float(item["value"]) for item in config["rewards"]
        }
        terminal_states = frozenset(
            tuple(item) for item in config.get("terminal_states", [])
        )
        start_state = tuple(config["start_state"]) if config.get("start_state") else None
        return cls(
            rows=int(config["rows"]),
            cols=int(config["cols"]),
            walls=walls,
            rewards=rewards,
            terminal_states=terminal_states,
            p_forward=float(config["p_forward"]),
            gamma=float(config["gamma"]),
            step_reward=float(config["step_reward"]),
            start_state=start_state,
        )

    @property
    def actions(self) -> tuple[Action, ...]:
        return tuple(ACTION_TO_DELTA.keys())

    def get_states(self) -> list[State]:
        return [
            (row, col)
            for row in range(self.rows)
            for col in range(self.cols)
            if (row, col) not in self.walls
        ]

    def get_non_terminal_states(self) -> list[State]:
        return [state for state in self.get_states() if not self.is_terminal(state)]

    def in_bounds(self, state: State) -> bool:
        row, col = state
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_wall(self, state: State) -> bool:
        return state in self.walls

    def is_terminal(self, state: State) -> bool:
        return state in self.terminal_states

    def get_actions(self, state: State) -> list[Action]:
        if self.is_wall(state) or self.is_terminal(state):
            return []
        return list(self.actions)

    def move(self, state: State, action: Action) -> State:
        if self.is_terminal(state):
            return state
        delta = ACTION_TO_DELTA[action]
        next_state = (state[0] + delta[0], state[1] + delta[1])
        if not self.in_bounds(next_state) or self.is_wall(next_state):
            return state
        return next_state

    def next_states_and_probs(self, state: State, action: Action) -> list[tuple[State, float]]:
        if self.is_terminal(state):
            return [(state, 1.0)]

        outcomes = [
            (self.move(state, action), self.p_forward),
            (self.move(state, LEFT_OF[action]), (1.0 - self.p_forward) / 2.0),
            (self.move(state, RIGHT_OF[action]), (1.0 - self.p_forward) / 2.0),
        ]
        merged: dict[State, float] = {}
        for next_state, probability in outcomes:
            merged[next_state] = merged.get(next_state, 0.0) + probability
        return sorted(merged.items())

    def reward(self, state: State) -> float:
        if self.is_terminal(state):
            return 0.0
        if state in self.rewards:
            return self.rewards[state]
        return self.step_reward

    def expected_action_value(
        self,
        state: State,
        action: Action,
        utilities: Dict[State, float],
    ) -> float:
        r = self.reward(state)
        total = 0.0
        for next_state, probability in self.next_states_and_probs(state, action):
            total += probability * utilities[next_state]
        return r + self.gamma * total

    def mean_utility(self, utilities: Dict[State, float]) -> float:
        states = self.get_states()
        return sum(utilities[state] for state in states) / float(len(states))
