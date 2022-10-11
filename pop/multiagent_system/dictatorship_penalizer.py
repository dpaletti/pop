#!/usr/bin/env ipython

from dataclasses import dataclass
from pop.multiagent_system.reward_distributor import Incentivizer
from typing import Dict, Hashable, List


class DictatorshipPenalizer:
    @dataclass
    class DictatorshipTracker:
        # TODO: here we assume that choices are non-negative and ranks too
        base_penalty_exponential_decay_half_life: float
        penalty_exponential_growth_factor: float
        smallest_base_penalty: float

        current_choice: int = -1
        repeated_choices: int = -1
        choice_rank: int = -1
        base_penalty: float = -1.0

        def choose(self, choice: int, choice_rank: int):
            if choice == self.current_choice:
                self.repeated_choices += 1
            else:
                self.current_choice = choice
                self.repeated_choices = 0
                self.choice_rank = choice_rank
                self.base_penalty = self._base_penalty(choice_rank)

        def dictatorship_penalty(self):
            return -Incentivizer._exponential_growth(
                self.base_penalty,
                self.penalty_exponential_growth_factor,
                self.repeated_choices,
            )

        def _base_penalty(self, choice_rank: int) -> float:
            return Incentivizer._exponential_decay(
                0,
                self.smallest_base_penalty,
                choice_rank,
                self.base_penalty_exponential_decay_half_life,
            )

        def reset(self):
            self.current_choice = -1
            self.repeated_choices = -1
            self.choice_rank = -1
            self.base_penalty = -1.0

    def __init__(
        self,
        choice_to_ranking: Dict[int, int],
        base_penalty_exponential_decay_half_life: float,
        penalty_exponential_growth_factor: float,
        smallest_base_penalty: float,
    ) -> None:

        self._choice_to_ranking = choice_to_ranking
        self._dictatorship_tracker = DictatorshipPenalizer.DictatorshipTracker(
            base_penalty_exponential_decay_half_life=base_penalty_exponential_decay_half_life,
            penalty_exponential_growth_factor=penalty_exponential_growth_factor,
            smallest_base_penalty=smallest_base_penalty,
        )

    def penalty(self, choice: int):
        self._dictatorship_tracker.choose(choice, self._choice_to_ranking[choice])
        return self._dictatorship_tracker.dictatorship_penalty()

    def add_choice(self, choice: int, ranking: int):
        self._choice_to_ranking[choice] = ranking

    def reset(self):
        self._dictatorship_tracker.reset()
