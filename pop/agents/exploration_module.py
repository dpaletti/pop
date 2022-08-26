import abc
from typing import Callable, Any
import numpy as np

from configs.agent_architecture import ExplorationParameters, EpsilonGreedyParameters


class ExplorationModule(abc.ABC):
    def __init__(self, exploration_parameters: ExplorationParameters, *args, **kwargs):
        ...

    @abc.abstractmethod
    def action_exploration(
        self, action_function: Callable[[Any], int]
    ) -> Callable[[Any], int]:
        ...

    @abc.abstractmethod
    def intrinsic_reward(self) -> float:
        ...


class EpsilonGreedy(ExplorationModule):
    def __init__(
        self, exploration_parameters: EpsilonGreedyParameters, number_of_actions: int
    ):
        super(EpsilonGreedy, self).__init__(exploration_parameters)
        self.max_epsilon = exploration_parameters.max_epsilon
        self.min_epsilon = exploration_parameters.min_epsilon
        self.exponential_half_life = int(exploration_parameters.epsilon_decay)
        self.decay_steps = 0
        self.number_of_actions = number_of_actions

    def action_exploration(
        self, action_function: Callable[[Any], int]
    ) -> Callable[[Any], int]:
        if np.random.rand() <= self._exponential_decay(
            self.max_epsilon,
            self.min_epsilon,
            self.exponential_half_life,
        ):
            return lambda x: np.random.choice(list(range(self.number_of_actions)))
        return action_function

    def intrinsic_reward(self) -> float:
        return 0

    def _exponential_decay(self, max_val: float, min_val: float, decay: int) -> float:
        return min_val + (max_val - min_val) * np.exp(-1.0 * self.decay_steps / decay)
