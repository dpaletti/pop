from collections import namedtuple
from typing import Optional
import numpy as np
from typing import Tuple, List, Any

Transition = namedtuple(
    "Transition", ("observation", "action", "next_observation", "reward", "done")
)


class ReplayMemory(object):
    """
    Prioritized Experience Replay buffer implemented on top of a structured Numpy Array.
    The idea is associating with every transition additional information:
    - priority: updated according to the loss
    - probability: computed out of transition priorities
    - weight: computed out of probabilities, correct for sampling bias
    Two hyper-parameters are introduced :math: `\alpha, \beta` which control how much
    we want to prioritize, at the end of the training we want to sample uniformly to
    avoid overfitting.
    Probability of sampling experience 'i':
    :math:`P(i)=\frac{p_i^\alpha}{\sum_{j=0}^N p_j^\alpha}` where :math:`p_i > 0` is
    the priority of transition 'i'.
    \alpha determines how much prioritization is used, with '\alpha > 0' corresponding
    to the uniform random sampling case.
    In order for the node_agents to converge the bias introduced by the non-uniform sampling needs
    to be corrected. We use importance sampling so that we can use weights when computing
    the loss:
    :math:`w_i = (\frac{1}{N}\frac{1}{P(i)})^\beta`
    :math:`beta` controls how strongly to correct for the bias where 0 means no correction while
    1 fully compensate for the bias. This weights are normalized by the maximum for stability
    concerns when computing the loss

    Attributes
    ----------
    capacity: ``int``
        capacity of the replay buffer

    memory: :class: np.array
        actual buffer implemented through a python deque

    alpha: ``float``
        strength of prioritized sampling

    buffer_length: ``int``
        number of transitions inside the memory

    random_state: :class: np.random
    """

    def __init__(
        self,
        capacity: int,
        alpha: float,
        random_state: Optional[np.random.RandomState] = None,
    ) -> None:
        self.memory = np.empty(
            capacity, dtype=[("priority", np.float32), ("transition", Transition)]
        )
        self.capacity = capacity
        self.alpha = alpha
        self.buffer_length = 0
        self.random_state = (
            np.random.RandomState() if random_state is None else random_state
        )

    def push(
        self,
        observation: Any,
        action: int,
        next_observation: Any,
        reward: float,
        done: bool,
    ) -> None:
        """Add a transition to the buffer together with its priority"""

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
        """Sample `batch_size` transitions"""
        priorities = self.memory[: self.buffer_length]["priority"]
        sampling_probabilities = priorities**self.alpha / np.sum(
            priorities**self.alpha
        )
        idxs = self.random_state.choice(
            np.arange(priorities.size),
            size=batch_size,
            replace=True,
            p=sampling_probabilities,
        )
        transitions = self.memory["transition"][idxs]
        weights = (self.buffer_length * sampling_probabilities[idxs]) ** -beta
        normalized_weights = weights / weights.max()

        return list(idxs), list(transitions), list(normalized_weights)

    def update_priorities(self, idxs: List[int], priorities: List[float]) -> None:
        """Update priorities of all samples with index in idxs"""
        self.memory["priority"][idxs] = priorities

    def __len__(self) -> int:
        return self.buffer_length
