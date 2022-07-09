from typing import Optional, Dict

import dgl
import networkx as nx
from grid2op.Environment import BaseEnv
from grid2op.Observation import BaseObservation

from agents.manager import Manager
from community_detection.community_detector import Community
from configs.architecture import Architecture
from multiagent_system.base_pop import BasePOP

from multiagent_system.space_factorization import EncodedAction


class DPOP(BasePOP):
    def __init__(
        self,
        env: BaseEnv,
        name: str,
        architecture: Architecture,
        training: bool,
        tensorboard_dir: Optional[str],
        checkpoint_dir: Optional[str],
        seed: int,
        device: Optional[str] = None,
    ):
        super(DPOP, self).__init__(
            env=env,
            name=name,
            architecture=architecture,
            training=training,
            tensorboard_dir=tensorboard_dir,
            checkpoint_dir=checkpoint_dir,
            seed=seed,
            device=device,
        )

        # Head Manager Initialization
        self.head_manager: Optional[Manager] = None

    def finalize_init_on_first_observation(
        self, first_observation: BaseObservation, first_observation_graph: nx.Graph
    ):
        super().finalize_init_on_first_observation(
            first_observation, first_observation_graph
        )
        self.head_manager: Manager = Manager.remote(
            agent_actions=self.node_number,
            node_features=3,
            architecture=self.architecture.head_manager,
            name="head_manager_" + self.name,
            training=self.training,
            device=self.device,
        )

    def get_action(self, graph: dgl.DGLHeteroGraph) -> EncodedAction:
        chosen_node: int = self.head_manager.remote().take_action(
            graph, mask=list(range(len(graph.nodes)))
        )
        return graph.nodes[chosen_node]["action"]

    def _extra_step(
        self,
        action: EncodedAction,
        reward: float,
        next_sub_graphs: Dict[Community, dgl.DGLHeteroGraph],
        next_graph: nx.Graph,
        done: bool,
    ):

        next_manager_actions: Dict[Community, EncodedAction] = self.get_manager_actions(
            next_sub_graphs
        )
        next_summarized_graph: dgl.DGLHeteroGraph = self.summarize_graph(
            next_graph, next_manager_actions, next_sub_graphs
        )

        self.head_manager.step.remote(
            observation=self.summarized_graph,
            action=action,
            reward=reward,
            next_observation=next_summarized_graph,
            done=done,
            stop_decay=False,
        )
