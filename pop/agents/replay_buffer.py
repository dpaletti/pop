from collections import namedtuple
import numpy as np
from typing import Tuple, List, Any

Transition = namedtuple(
    "Transition", ("observation", "action", "next_observation", "reward", "done")
)


class ReplayMemory(object):
    def __init__(
        self,
        capacity: int,
        alpha: float,
    ) -> None:
        self.memory = np.empty(
            capacity, dtype=[("priority", np.float32), ("transition", Transition)]
        )
        self.capacity = capacity
        self.alpha = alpha
        self.buffer_length = 0

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

    def sample(
        self, batch_size: int, beta: float
    ) -> Tuple[List[int], List[Transition], List[float]]:

        priorities = self.memory[: self.buffer_length]["priority"]
        sampling_probabilities = priorities**self.alpha / np.sum(
            priorities**self.alpha
        )

        indices = np.random.choice(
            np.arange(priorities.size),
            size=batch_size,
            replace=True,
            p=sampling_probabilities,
        )
        transitions = self.memory["transition"][indices]
        weights = (self.buffer_length * sampling_probabilities[indices]) ** -beta
        normalized_weights = weights / weights.max()

        return list(indices), list(transitions), list(normalized_weights)

    def update_priorities(self, idxs: List[int], priorities: List[float]) -> None:
        self.memory["priority"][idxs] = priorities

    def __len__(self) -> int:
        return self.buffer_length
