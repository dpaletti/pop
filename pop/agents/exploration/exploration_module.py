import abc
import functools
from typing import Any, Callable, Dict

import dgl


class ExplorationModule(abc.ABC):
    @abc.abstractmethod
    def __init__(self, agent):
        ...

    @abc.abstractmethod
    def update(self, *args, **kwargs) -> None:
        ...

    def action_exploration(self, action_function: Callable) -> Callable:
        @functools.wraps(action_function)
        def wrap(wrapped_self, transformed_observation: dgl.DGLHeteroGraph, **kwargs):
            return action_function(transformed_observation, kwargs.get("mask"))

        return wrap

    def compute_intrinsic_reward(
        self,
        observation: dgl.DGLHeteroGraph,
        next_observation: dgl.DGLHeteroGraph,
        action: int,
        done: bool,
    ) -> float:
        return 0

    @abc.abstractmethod
    def get_state(self) -> Dict[str, Any]:
        ...

    @staticmethod
    @abc.abstractmethod
    def load_state(state: Dict[str, Any]):
        ...

    @abc.abstractmethod
    def get_state_to_log(self) -> Dict[str, Any]:
        ...

    def apply_intrinsic_reward(
        self, step_function: Callable, intrinsic_reward_relevance: float
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
                observation, next_observation, action, done
            )
            return step_function(
                observation,
                action,
                reward + intrinsic_reward_relevance * intrinsic_reward,
                next_observation,
                done,
                stop_decay=stop_decay,
            )

        return wrap
