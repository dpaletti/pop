import abc
import functools
from typing import Callable

import dgl


class ExplorationModule(abc.ABC):
    @abc.abstractmethod
    def update(self, *args, **kwargs) -> None:
        ...

    @abc.abstractmethod
    def action_exploration(self, action_function: Callable) -> Callable:
        ...

    @abc.abstractmethod
    def compute_intrinsic_reward(
        self,
        observation: dgl.DGLHeteroGraph,
        next_observation: dgl.DGLHeteroGraph,
        action: int,
    ) -> float:
        ...

    def apply_intrinsic_reward(
        self,
        step_function: Callable,
    ) -> Callable:
        @functools.wraps(step_function)
        def wrap(
            wrapped_self,
            observation,
            action,
            reward,
            next_observation,
            done,
            stop_decay,
        ):
            intrinsic_reward = self.compute_intrinsic_reward(
                observation, next_observation, action
            )
            return step_function(
                observation,
                action,
                intrinsic_reward + reward,
                next_observation,
                done,
                stop_decay=stop_decay,
            )

        return wrap
