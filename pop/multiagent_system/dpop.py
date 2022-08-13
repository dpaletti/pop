from typing import Optional, Dict, Any, List

import dgl
import networkx as nx
from grid2op.Environment import BaseEnv
from grid2op.Observation import BaseObservation
from ray.util.client import ray

from agents.manager import Manager
from agents.ray_gcn_agent import RayGCNAgent
from agents.ray_shallow_gcn_agent import RayShallowGCNAgent
from community_detection.community_detector import Community
from configs.architecture import Architecture
from multiagent_system.base_pop import BasePOP

from multiagent_system.space_factorization import EncodedAction, Substation


class DPOP(BasePOP):
    def __init__(
        self,
        env: BaseEnv,
        name: str,
        architecture: Architecture,
        training: bool,
        seed: int,
        checkpoint_dir: Optional[str] = None,
        tensorboard_dir: Optional[str] = None,
        device: Optional[str] = None,
        pre_initialized: bool = False,
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
            pre_initialized=pre_initialized,
        )

        # Head Manager Initialization
        self.head_manager: Optional[Manager] = None

        # Initialize Ray
        ray.init(local_mode=True)

    def _finalize_init_on_first_observation(
        self,
        first_observation: BaseObservation,
        first_observation_graph: nx.Graph,
        pre_initialized=False,
    ):

        super()._finalize_init_on_first_observation(
            first_observation, first_observation_graph, pre_initialized=pre_initialized
        )
        if not self.pre_initialized:
            # WARNING: assuming that all managers have the
            self.head_manager: Manager = Manager.remote(
                agent_actions=self.env.n_sub,
                node_features=int(
                    ray.get(
                        list(self.community_to_manager.values())[
                            0
                        ].get_embedding_size.remote()
                    )
                )
                + self.env.n_sub
                + 1,  # Manager Node Embedding + Manager Community (1 hot encoded) + selected action
                architecture=self.architecture.head_manager,
                name="head_manager_" + self.name,
                training=self.training,
                device=str(self.device),
            )

    def get_action(self, graph: dgl.DGLHeteroGraph) -> int:
        chosen_node: int = int(
            ray.get(
                self.head_manager.take_action.remote(
                    graph, mask=list(range(graph.num_nodes()))
                )
            )
        )
        return chosen_node

    def _extra_step(
        self,
        action: int,
        reward: float,
        next_sub_graphs: Dict[Community, dgl.DGLHeteroGraph],
        next_substation_to_encoded_action: Dict[Substation, EncodedAction],
        next_graph: nx.Graph,
        done: bool,
        next_communities: List[Community],
        next_community_to_manager: Dict[Community, Manager],
    ):
        try:
            next_community_to_substation: Dict[
                Community, Substation
            ] = self._get_manager_actions(
                next_sub_graphs,
                next_substation_to_encoded_action,
                next_communities,
                next_community_to_manager,
            )

            next_summarized_graph: dgl.DGLHeteroGraph = self._compute_summarized_graph(
                next_graph,
                next_sub_graphs,
                next_substation_to_encoded_action,
                next_community_to_substation,
                new_communities=next_communities,
                new_community_to_manager_dict=next_community_to_manager,
            )

            loss = ray.get(
                self.head_manager.step.remote(
                    observation=self.summarized_graph,
                    action=action,
                    reward=reward,
                    next_observation=next_summarized_graph,
                    done=done,
                    stop_decay=False,
                )
            )

            if loss is not None:
                self.log_loss(
                    {ray.get(self.head_manager.get_name.remote()): loss},
                    self.train_steps,
                )
        except KeyError as e:
            print("...")
            raise e

    def get_state(self: "DPOP") -> Dict[str, Any]:
        state: Dict[str, Any] = super().get_state()
        state["head_manager_state"] = ray.get(self.head_manager.get_state.remote())
        return state

    @staticmethod
    def factory(
        checkpoint: Dict[str, Any],
        env: Optional[BaseEnv] = None,
        tensorboard_dir: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        name: Optional[str] = None,
        training: Optional[bool] = None,
    ) -> "DPOP":
        dpop: "DPOP" = DPOP(
            env=env,
            name=checkpoint["name"] if name is None else name,
            architecture=Architecture(load_from_dict=checkpoint["architecture"]),
            training=training,
            tensorboard_dir=tensorboard_dir,
            checkpoint_dir=checkpoint_dir,
            seed=checkpoint["seed"],
            device=checkpoint["device"],
        )
        dpop.pre_initialized = True
        dpop.alive_steps = checkpoint["alive_steps"]
        dpop.episodes = checkpoint["episodes"]
        dpop.train_steps = checkpoint["train_steps"]
        dpop.edge_features = checkpoint["edge_features"]
        dpop.node_features = checkpoint["node_features"]
        dpop.substation_to_agent = {
            sub_id: RayGCNAgent.load(checkpoint_dict=agent_state)
            if "optimizer_state" in list(agent_state.keys())
            else RayShallowGCNAgent.load(checkpoint_dict=agent_state)
            for sub_id, agent_state in checkpoint["agents_state"].items()
        }
        dpop.community_to_manager = {
            community: Manager.load(checkpoint_dict=manager_state)
            for community, manager_state in checkpoint["managers_state"].items()
        }
        dpop.head_manager = Manager.load(
            checkpoint_dict=checkpoint["head_manager_state"]
        )
        dpop.communities = checkpoint["communities"]
        return dpop
