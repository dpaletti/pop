from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Tuple

from math import exp, log10


class Incentivizer:
    @dataclass
    class ElectionHistory:
        consecutive_elections: int = 0
        consecutive_rejections: int = 0

        def elect(self):
            self.consecutive_elections += 1
            self.consecutive_rejections = 0

        def reject(self):
            self.consecutive_rejections += 1
            self.consecutive_elections = 0

        def reset(self):
            self.consecutive_elections = 0
            self.consecutive_rejections = 0

    @dataclass
    class Incentive:
        minimum_penalty: float
        minimum_prize: float
        penalty_growth_rate: float
        prize_growth_rate: float

        def incentive(self, election_history: "Incentivizer.ElectionHistory") -> float:
            if election_history.consecutive_elections > 0:
                return self._prize(election_history.consecutive_elections)
            return self._penalty(election_history.consecutive_rejections)

        def _penalty(self, consecutive_rejections: int):
            return -Incentivizer._exponential_growth(
                self.minimum_penalty,
                self.penalty_growth_rate,
                consecutive_rejections - 1,
            )

        def _prize(self, consecutive_elections: int):
            return Incentivizer._logarithmic_growth(
                self.minimum_prize, self.prize_growth_rate, consecutive_elections - 1
            )

    def __init__(
        self,
        agent_actions: Dict[Hashable, int],
        largest_base_prize: float,
        smallest_base_penalty: float,
        prize_logarithmic_growth_factor: float,
        penalty_exponential_growth_factor: float,
        base_prize_exponential_decay_half_life: float,
        base_penalty_exponential_growth_factor: float,
    ) -> None:
        # State does not need to be loaded nor saved
        # Elections are reset at every episode

        self._agents: List[Hashable] = list(agent_actions.keys())
        self.agent_actions = agent_actions
        self._rank_to_agents: Dict[int, List[Hashable]] = self._rank_agents(
            self.agent_actions
        )
        self._elections: Dict[Hashable, Incentivizer.ElectionHistory] = {
            agent: Incentivizer.ElectionHistory() for agent in self._agents
        }

        self._smallest_penalty: float = smallest_base_penalty
        self._largest_prize: float = largest_base_prize
        self._prize_logarithmic_growth_factor: float = prize_logarithmic_growth_factor
        self._penalty_exponential_growth_factor: float = (
            penalty_exponential_growth_factor
        )
        self._base_prize_exponential_decay_factor: float = (
            base_prize_exponential_decay_half_life
        )
        self._base_penalty_exponential_growth_factor: float = (
            base_penalty_exponential_growth_factor
        )

        self._agents_incentives: Dict[
            Hashable, "Incentivizer.Incentive"
        ] = self._compute_base_incentives(
            agent_ranking=self._rank_to_agents,
            smallest_penalty=self._smallest_penalty,
            largest_prize=self._largest_prize,
            prize_exponential_decay_factor=base_prize_exponential_decay_half_life,
            penalty_exponential_growth_factor=base_penalty_exponential_growth_factor,
        )

    def incentives(
        self,
        elected_agents: List[Hashable],
    ) -> Dict[Hashable, float]:

        for agent in self._agents:
            if agent in elected_agents:
                self._elections[agent].elect()
            else:
                self._elections[agent].reject()

        return {
            agent: self._agents_incentives[agent].incentive(self._elections[agent])
            for agent in self._agents
        }

    def add_agent(self, agent_to_add: Hashable, actions: int):
        self._agents.append(agent_to_add)
        self.agent_actions[agent_to_add] = actions
        self._rank_to_agents = self._rank_agents(self.agent_actions)
        self._elections = {
            agent: Incentivizer.ElectionHistory() for agent in self._agents
        }
        self._agents_incentives: Dict[
            Hashable, "Incentivizer.Incentive"
        ] = self._compute_base_incentives(
            agent_ranking=self._rank_to_agents,
            smallest_penalty=self._smallest_penalty,
            largest_prize=self._largest_prize,
            prize_exponential_decay_factor=self._base_prize_exponential_decay_factor,
            penalty_exponential_growth_factor=self._base_penalty_exponential_growth_factor,
        )

    def reset(self):
        for election_history in self._elections.values():
            election_history.reset()

    def _invert_ranking(
        self, ranking: Dict[int, List[Hashable]]
    ) -> Dict[Hashable, List[int]]:
        return {
            agent: rank for rank, agents in ranking.items() for agent in agents
        }  # type: ignore

    def _compute_base_incentives(
        self,
        agent_ranking: Dict[int, List[Hashable]],
        largest_prize: float,
        smallest_penalty: float,
        prize_exponential_decay_factor: float,
        penalty_exponential_growth_factor: float,
    ) -> Dict[Hashable, "Incentivizer.Incentive"]:

        incentives: Dict[Hashable, "Incentivizer.Incentive"] = {}
        for rank, agents in agent_ranking.items():
            base_penalty: float = Incentivizer._exponential_growth(
                smallest_penalty, penalty_exponential_growth_factor, rank
            )
            base_prize: float = Incentivizer._exponential_decay(
                0, largest_prize, rank, prize_exponential_decay_factor
            )
            for agent in agents:
                incentives[agent] = Incentivizer.Incentive(
                    minimum_penalty=base_penalty,
                    minimum_prize=base_prize,
                    penalty_growth_rate=self._penalty_exponential_growth_factor,
                    prize_growth_rate=self._prize_logarithmic_growth_factor,
                )
        return incentives

    @staticmethod
    def _rank_agents(agent_actions: Dict[Hashable, int]) -> Dict[int, List[Hashable]]:
        full_action_space = sum(list(agent_actions.values()))
        ranking: Dict[int, List[Hashable]] = {}
        ordered_action_spaces: Dict[float, List[Hashable]] = {}
        for agent, action_space_size in agent_actions.items():
            action_space_portion = full_action_space / action_space_size
            if ordered_action_spaces.get(action_space_portion):
                ordered_action_spaces[action_space_portion].append(agent)
            else:
                ordered_action_spaces[action_space_portion] = [agent]

        ordered_action_spaces = dict(
            sorted(ordered_action_spaces.items(), reverse=True)
        )
        for position, (action_space_portion, agents) in enumerate(
            ordered_action_spaces.items()
        ):
            ranking[position] = agents

        return ranking

    @staticmethod
    def _exponential_decay(
        initial_value: float, target_value: float, steps: int, half_life: float
    ) -> float:
        return initial_value + (target_value - initial_value) * exp(
            -1.0 * steps / half_life
        )

    @staticmethod
    def _exponential_growth(
        initial_value: float, growth_rate: float, steps: int
    ) -> float:
        return initial_value * (1 + growth_rate) ** steps

    @staticmethod
    def _logarithmic_growth(
        initial_value: float, growth_rate: float, steps: int
    ) -> float:
        return growth_rate * log10(steps + 1) + initial_value
