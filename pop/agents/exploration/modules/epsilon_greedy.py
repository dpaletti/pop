import functools
from typing import cast, Any, Callable, Optional, List

import dgl
import numpy as np

from agents.base_gcn_agent import BaseGCNAgent
from agents.exploration.exploration_module import ExplorationModule
from configs.agent_architecture import EpsilonGreedyParameters


class EpsilonGreedy(ExplorationModule):
    def __init__(self, agent: BaseGCNAgent):
        exploration_parameters: EpsilonGreedyParameters = cast(
            EpsilonGreedyParameters, agent.architecture.exploration
        )
        self.max_epsilon = exploration_parameters.max_epsilon
        self.min_epsilon = exploration_parameters.min_epsilon
        self.exponential_half_life = int(exploration_parameters.epsilon_decay)
        self.decay_steps = 0
        self.number_of_actions = agent.actions

    def update(self, *args, **kwargs) -> None:
        self.decay_steps += 1

    def action_exploration(self, action_function: Callable) -> Callable:
        @functools.wraps(action_function)
        def wrap(wrapped_self, transformed_observation: dgl.DGLHeteroGraph, **kwargs):
            action_list = list(range(self.number_of_actions))
            mask: Optional[List[int]] = kwargs.get("mask")

            if np.random.rand() <= self._exponential_decay(
                self.max_epsilon,
                self.min_epsilon,
                self.exponential_half_life,
            ):
                return np.random.choice(
                    action_list,
                    p=[1 / len(mask) if action in mask else 0 for action in action_list]
                    if mask
                    else None,
                )
            return action_function(transformed_observation, mask)

        return wrap

    def compute_intrinsic_reward(
        self,
        observation: dgl.DGLHeteroGraph,
        next_observation: dgl.DGLHeteroGraph,
        action: int,
    ) -> float:
        return 0

    def _exponential_decay(self, max_val: float, min_val: float, decay: int) -> float:
        return min_val + (max_val - min_val) * np.exp(-1.0 * self.decay_steps / decay)
