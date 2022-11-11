from pop.agents.base_gcn_agent import BaseGCNAgent
from typing import Dict, Any

from pop.agents.exploration.modules.episodic_memory import EpisodicMemory
from pop.agents.exploration.modules.epsilon_greedy import EpsilonGreedy


class EpsilonEpisodic(EpisodicMemory, EpsilonGreedy):
    def __init__(self, agent: BaseGCNAgent):
        super().__init__(agent)

    def update(self, *args, **kwargs) -> None:
        EpisodicMemory.update(self, *args, **kwargs)
        EpsilonGreedy.update(self, *args, **kwargs)

    def get_state(self):
        return {**EpisodicMemory.get_state(self), **EpsilonGreedy.get_state(self)}

    def get_state_to_log(self) -> Dict[str, Any]:
        return {
            **EpisodicMemory.get_state_to_log(self),
            **EpsilonGreedy.get_state_to_log(self),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        EpisodicMemory.load_state(self, state)
        EpsilonGreedy.load_state(self, state)
