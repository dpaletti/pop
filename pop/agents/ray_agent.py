from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Tuple, List, Optional

from torch import Tensor

from networks.dueling_net import DuelingNet


class RayAgent(ABC):
    @abstractmethod
    def get_q_network(self) -> Optional[DuelingNet]:
        ...

    @abstractmethod
    def reset_decay(self):
        ...

    @abstractmethod
    def load_state(
        self,
        optimizer_state: dict,
        q_network_state: OrderedDict[str, Tensor],
        target_network_state: OrderedDict[str, Tensor],
        losses: List[float],
        actions: List[int],
        decay_steps: int,
        alive_steps: int,
        train_steps: int,
        learning_steps: int,
        reset_decay=False,
    ) -> None:
        ...

    @abstractmethod
    def get_name(self) -> str:
        ...
