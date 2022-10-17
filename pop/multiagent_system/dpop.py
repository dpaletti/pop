from typing import Any, Dict, List, Optional

import dgl
import networkx as nx
from pop.configs import architecture
import psutil
from grid2op.Environment import BaseEnv
from ray.util.client import ray
from tqdm import tqdm

from pop.agents.manager import Manager
from pop.agents.ray_gcn_agent import RayGCNAgent
from pop.agents.ray_shallow_gcn_agent import RayShallowGCNAgent
from pop.community_detection.community_detector import Community
from pop.configs.architecture import Architecture
from pop.multiagent_system.base_pop import BasePOP
from pop.multiagent_system.space_factorization import EncodedAction, Substation

from pop.multiagent_system.dictatorship_penalizer import DictatorshipPenalizer


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
    ):
        process = psutil.Process()
        process.cpu_affinity(list(range(0, 14)))
        print("Running on cores: " + str(process.cpu_affinity()))
        ray.init(local_mode=False, num_cpus=len(process.cpu_affinity()) * 2)
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
        try:
            node_features = self.architecture.manager.embedding.layers[-1].kwargs[
                "out_feats"
            ]
        except KeyError:
            # If last layer is an activation one
            # Should make this general
            node_features = self.architecture.manager.embedding.layers[-2].kwargs[
                "out_feats"
            ]

        # Head Manager Initialization
        self.head_manager: Optional[Manager] = Manager.remote(
            agent_actions=self.env.n_sub * 2,
            node_features=[self.architecture.pop.head_manager_embedding_name],
            architecture=self.architecture.head_manager,
            name="head_manager_" + self.name,
            training=self.training,
            device=str(self.device),
            single_node_features=int(node_features)
            + self.env.n_sub * 2
            + 1,  # Manager Node Embedding + Manager Community (1 hot encoded) + selected action
        )
        if self.architecture.pop.dictatorship_penalty:
            self.dictatorship_penalty = DictatorshipPenalizer(
                {choice: 1 for choice in range(self.env.n_sub)},
                **self.architecture.pop.dictatorship_penalty
            )

    def get_action(self, graph: dgl.DGLHeteroGraph) -> int:
        chosen_node: int = int(
            ray.get(
                self.head_manager.take_action.remote(
                    graph, mask=list(range(graph.num_nodes()))
                )
            )
        )

        self.log_exploration(
            "head_manager",
            ray.get(self.head_manager.get_exploration_logs.remote()),
            self.train_steps,
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
        penalty = 0
        if self.architecture.pop.dictatorship_penalty:
            penalty = self.dictatorship_penalty.penalty(action)

        if done:
            loss, full_reward = ray.get(
                self.head_manager.step.remote(
                    observation=self.summarized_graph,
                    action=action,
                    reward=reward + penalty,
                    next_observation=dgl.DGLGraph(),
                    done=done,
                    stop_decay=False,
                )
            )
        else:
            next_community_to_substation: Dict[
                Community, Substation
            ] = self._get_manager_actions(
                next_graph,
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

            loss, full_reward = ray.get(
                self.head_manager.step.remote(
                    observation=self.summarized_graph,
                    action=action,
                    reward=reward + penalty,
                    next_observation=next_summarized_graph,
                    done=done,
                    stop_decay=False,
                )
            )

        self.log_step(
            losses=[loss],
            implicit_rewards=[full_reward - reward],
            names=[self.name],
            train_steps=self.train_steps,
            dictatorship_penalties=[penalty],
        )

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
        dpop.alive_steps = checkpoint["alive_steps"] if training else 0
        dpop.episodes = checkpoint["episodes"] if training else 0
        dpop.train_steps = checkpoint["train_steps"] if training else 0
        dpop.edge_features = checkpoint["edge_features"]
        dpop.node_features = checkpoint["node_features"]
        dpop.manager_initialization_threshold = checkpoint[
            "manager_initialization_threshold"
        ]
        dpop.community_update_steps = checkpoint["community_update_steps"]
        dpop.substation_to_agent = {
            sub_id: RayGCNAgent.load(
                checkpoint=agent_state,
                training=training,
            )
            if "optimizer_state" in list(agent_state.keys())
            else RayShallowGCNAgent.load(checkpoint=agent_state)
            for sub_id, agent_state in tqdm(checkpoint["agents_state"].items())
        }
        print("Loading Managers")
        dpop.managers_history = {
            Manager.load(
                checkpoint=manager_state,
                training=training,
            ): history
            for _, (manager_state, history) in tqdm(
                checkpoint["managers_state"].items()
            )
        }
        # TODO: managers here are treated as equal, they may be not

        if dpop.architecture.pop.incentives:
            for manager in dpop.managers_history.keys():
                dpop.manager_incentives.add_agent(manager, 1)

        if dpop.architecture.pop.dictatorship_penalty:
            for manager in dpop.managers_history.keys():
                dpop.manager_dictatorship_penalties[manager] = DictatorshipPenalizer(
                    choice_to_ranking={
                        substation: len(action_converter.all_actions)
                        for substation, action_converter in dpop.substation_to_action_converter.items()
                    },
                    **dpop.architecture.pop.dictatorship_penalty
                )

        print("Loading Head Manager")
        dpop.head_manager = Manager.load(
            checkpoint=checkpoint["head_manager_state"],
            training=training,
        )
        print("Loading is over")
        return dpop
