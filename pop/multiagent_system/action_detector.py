from collections import deque
from typing import List


class ActionDetector:
    def __init__(
        self,
        loop_length: int = 1,
        penalty_value: float = 0,
        repeatable_actions: List[int] = [],
    ):
        self.loop_length: int = loop_length
        self.penalty_value: float = penalty_value
        self.penalize: int = 0
        self.repeatable_actions: List[int] = repeatable_actions
        if self.loop_length < 1:
            return

        self.action_memory: deque = deque([-1] * loop_length, loop_length)

    def is_repeated(self, action: int) -> bool:
        if self.loop_length < 1:
            return False

        if not action in self.repeatable_actions and action in self.action_memory:
            self.penalize += 1
            self.action_memory.append(action)
            return True

        self.penalize = 0
        self.action_memory.append(action)
        return False

    def penalty(self) -> float:
        if self.penalize:
            return -self.penalize * self.penalty_value
        return 0
