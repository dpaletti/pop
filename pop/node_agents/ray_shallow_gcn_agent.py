from typing import Union, Optional, Tuple

import networkx as nx
import torch as th
import ray
from dgl import DGLHeteroGraph
from torch import Tensor

from node_agents.base_gcn_agent import BaseGCNAgent


@ray.remote
class RayShallowGCNAgent(BaseGCNAgent):
    def __init__(
        self,
        agent_actions: int,
        node_features: int,
        edge_features: int,
        architecture: Union[str, dict],
        name: str,
        training: bool,
        device: str,
    ):
        BaseGCNAgent.__init__(
            self,
            agent_actions=agent_actions,
            node_features=node_features,
            edge_features=edge_features,
            architecture=architecture,
            name=name,
            training=training,
            device=device,
        )

        # Logging
        self.losses = []
        self.actions_taken = []

    def get_q_network(self):
        raise Exception("Shallow agent: " + self.name + " has no q_network")

    def get_state(self):
        return [
            dict(),
            dict(),
            dict(),
            0,
            self.actions_taken,
            0,
            self.alive_steps,
            self.trainsteps,
            self.learning_steps,
        ]

    def get_name(self):
        return self.name

    def load_state(
        self, losses, actions, alive_steps, trainsteps, learning_steps, **kwargs
    ):
        self.losses = losses
        self.actions_taken = actions
        self.decay_steps = 0
        self.alive_steps = alive_steps
        self.trainsteps = trainsteps
        self.learning_steps = learning_steps

    def take_action(
        self,
        transformed_observation: DGLHeteroGraph,
    ) -> Tuple[int, float]:
        return 0, 0  # Action chosen always zero and Epsilon always zero

    def step(
        self,
        observation,
        action: int,
        reward: float,
        next_observation: nx.Graph,
        done: bool,
        stop_decay: bool = False,
    ):

        if done:
            self.episodes += 1
            self.trainsteps += 1

        else:
            self.trainsteps += 1
            self.alive_steps += 1
            if not stop_decay:
                self.decay_steps += 1

            # every so often the node_agents should learn from experiences
            if self.trainsteps % self.architecture["learning_frequency"] == 0:
                self.learning_steps += 1
                return th.tensor(0.0, requires_grad=True), None, None
            return None, None, None
