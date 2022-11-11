import functools
from typing import cast, Any, Callable, Optional, List, Dict

import dgl
import numpy as np

from agents.base_gcn_agent import BaseGCNAgent
from agents.exploration.exploration_module import ExplorationModule
from configs.agent_architecture import EpsilonGreedyParameters


class EpsilonGreedy(ExplorationModule):
    def __init__(self, agent: BaseGCNAgent):
        super().__init__(agent)
        exploration_parameters: EpsilonGreedyParameters = cast(
            EpsilonGreedyParameters, agent.architecture.exploration
        )
        self.max_epsilon = exploration_parameters.max_epsilon
        self.min_epsilon = exploration_parameters.min_epsilon
        self.exponential_half_life = int(exploration_parameters.epsilon_decay)
        self.decay_steps = 0
        self.number_of_actions = agent.actions
        self.epsilon: float = 0

    def update(self, *args, **kwargs) -> None:
        self.decay_steps += 1

    def action_exploration(self, action_function: Callable) -> Callable:
        @functools.wraps(action_function)
        def wrap(wrapped_self, transformed_observation: dgl.DGLHeteroGraph, **kwargs):
            action_list = list(range(self.number_of_actions))
            mask: Optional[List[int]] = kwargs.get("mask")

            self.epsilon = self._exponential_decay(
                self.max_epsilon,
                self.min_epsilon,
                self.exponential_half_life,
            )
            if np.random.rand() <= self.epsilon:
                return np.random.choice(
                    action_list,
                    p=[1 / len(mask) if action in mask else 0 for action in action_list]
                    if mask
                    else None,
                )
            return action_function(transformed_observation, mask)

        return wrap

    def _exponential_decay(self, max_val: float, min_val: float, decay: int) -> float:
        return min_val + (max_val - min_val) * np.exp(-1.0 * self.decay_steps / decay)

    def get_state(self) -> Dict[str, Any]:
        return {"decay_steps": self.decay_steps}

    def load_state(self, state: Dict[str, Any]) -> None:
        self.decay_steps = state["decay_steps"]

    def get_state_to_log(self) -> Dict[str, Any]:
        return {"epsilon": self.epsilon}
