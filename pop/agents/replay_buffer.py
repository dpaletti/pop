from collections import namedtuple
from dataclasses import asdict

import numpy as np
from typing import Dict, Tuple, List, Any
import pandas as pd

from pop.configs.agent_architecture import ReplayMemoryParameters

Transition = namedtuple(
    "Transition", ("observation", "action", "next_observation", "reward", "done")
)


class ReplayMemory(object):
    def __init__(self, architecture: ReplayMemoryParameters) -> None:
        self.capacity = architecture.capacity
        self.memory = np.empty(
            self.capacity, dtype=[("priority", np.float32), ("transition", Transition)]
        )
        self.architecture: ReplayMemoryParameters = architecture
        self.alpha: float = architecture.alpha
        self.buffer_length: int = 0
        self.steps: int = 0
        self.max_beta: float = architecture.max_beta
        self.min_beta: float = architecture.min_beta
        self.half_life: float = architecture.beta_decay
        self.beta: float = self.max_beta

    def push(
        self,
        observation: Any,
        action: int,
        next_observation: Any,
        reward: float,
        done: bool,
    ) -> None:

        transition = Transition(
            observation=observation,
            action=action,
            next_observation=next_observation,
            reward=reward,
            done=done,
        )

        # Assign priority to current transition
        priority = 1.0 if self.is_empty() else self.memory["priority"].max()

        if self.is_full():
            if priority > self.memory["priority"].min():
                # Replace the lowest priority transition
                idx = self.memory["priority"].argmin()
                self.memory[idx] = (priority, transition)
        else:
            # Add to the buffer
            self.memory[self.buffer_length] = (priority, transition)
            self.buffer_length += 1

    def is_empty(self) -> bool:
        return self.buffer_length == 0

    def is_full(self) -> bool:
        return self.buffer_length == self.capacity

    def update(self):
        self.steps += 1
        self.beta = self.min_beta + (self.max_beta - self.min_beta) * np.exp(
            -1.0 * self.steps / self.half_life
        )

    def sample(
        self, batch_size: int, epsilon: float = 1e-4
    ) -> Tuple[List[int], List[Transition], List[float]]:

        priorities = self.memory[: self.buffer_length]["priority"]
        sampling_probabilities = priorities**self.alpha / (
            np.sum(priorities**self.alpha) + epsilon
        )

        indices = np.random.choice(
            np.arange(priorities.size),
            size=batch_size,
            replace=True,
            p=sampling_probabilities,
        )
        transitions = self.memory["transition"][indices]
        weights = (self.buffer_length * sampling_probabilities[indices]) ** -self.beta
        normalized_weights = weights / weights.max()

        return list(indices), list(transitions), list(normalized_weights)

    def update_priorities(self, idxs: List[int], priorities: List[float]) -> None:
        self.memory["priority"][idxs] = priorities

    def __len__(self) -> int:
        return self.buffer_length

    def get_state(self) -> dict:
        return {
            "architecture": asdict(self.architecture),
            "beta": self.beta,
            "buffer_length": self.buffer_length,
            "memory": pd.DataFrame(self.memory).to_dict(),
        }

    def load_state(self, state_dict: dict) -> None:
        self.buffer_length = state_dict["buffer_length"]
        self.beta = state_dict["beta"]

        for idx, (priority, transition) in enumerate(
            zip(
                state_dict["memory"]["priority"].values(),
                state_dict["memory"]["transition"].values(),
            )
        ):
            self.memory[idx] = (priority, transition)
