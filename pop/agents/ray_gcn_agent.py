from typing import Dict, Any, Optional

import ray
from ray import ObjectRef

from pop.agents.base_gcn_agent import BaseGCNAgent
from pop.configs.agent_architecture import AgentArchitecture
from pop.networks.dueling_net import DuelingNet

import logging
import warnings
logging.getLogger("lightning").addHandler(logging.NullHandler())
logging.getLogger("lightning").propagate = False

warnings.filterwarnings("ignore", category=UserWarning)


@ray.remote(num_cpus=1, scheduling_strategy="SPREAD")
class RayGCNAgent(BaseGCNAgent):
    def __init__(
        self,
        agent_actions: int,
        node_features: int,
        architecture: AgentArchitecture,
        name: str,
        training: bool,
        device: str,
        edge_features: Optional[int] = None,
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
            tensorboard_dir=None,
            log_dir=None,
        )

    def get_q_network(self) -> DuelingNet:
        return self.q_network

    def get_name(self) -> str:
        return self.name

    def reset_decay(self):
        self.decay_steps = 0

    @staticmethod
    def factory(checkpoint: Dict[str, Any], **kwargs) -> ObjectRef:
        agent: ObjectRef = RayGCNAgent.remote(
            agent_actions=checkpoint["agent_actions"],
            node_features=checkpoint["node_features"],
            architecture=AgentArchitecture(load_from_dict=checkpoint["architecture"]),
            name=checkpoint["name"],
            training=bool(kwargs.get("training")),
            device=checkpoint["device"],
            edge_features=checkpoint["edge_features"],
        )
        agent.load_state.remote(
            optimizer_state=checkpoint["optimizer_state"],
            q_network_state=checkpoint["q_network_state"],
            target_network_state=checkpoint["target_network_state"],
            memory=checkpoint["memory"],
            exploration=checkpoint["exploration"],
            alive_steps=checkpoint["alive_steps"],
            train_steps=checkpoint["train_steps"],
            learning_steps=checkpoint["learning_steps"],
        )
        return agent
