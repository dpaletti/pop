import logging
import warnings
from typing import Any, Dict, Optional, Tuple

import ray
from ray import ObjectRef
import torch as th

from pop.agents.base_gcn_agent import BaseGCNAgent
from pop.configs.agent_architecture import AgentArchitecture
from pop.networks.dueling_net import DuelingNet

import logging
import warnings

from pop.constants import PER_PROCESS_GPU_MEMORY_FRACTION

logging.getLogger("lightning").addHandler(logging.NullHandler())
logging.getLogger("lightning").propagate = False

warnings.filterwarnings("ignore", category=UserWarning)


@ray.remote(
    num_cpus=0.5,
    num_gpus=0 if not th.cuda.is_available() else PER_PROCESS_GPU_MEMORY_FRACTION,
)
class RayGCNAgent(BaseGCNAgent):
    def __init__(
        self,
        agent_actions: int,
        node_features: int,
        architecture: AgentArchitecture,
        name: str,
        training: bool,
        device: str,
        feature_ranges: Dict[str, Tuple[float, float]],
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
            feature_ranges=feature_ranges,
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
            architecture=AgentArchitecture(load_from_dict=checkpoint["architecture"])
            if kwargs.get("architecture") is None
            else kwargs["architecture"],
            name=checkpoint["name"],
            training=bool(kwargs.get("training")),
            device=checkpoint["device"],
            edge_features=checkpoint["edge_features"],
            feature_ranges=checkpoint["feature_ranges"],
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
            reset_exploration=kwargs["reset_exploration"],
        )
        return agent
