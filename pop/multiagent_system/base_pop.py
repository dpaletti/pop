from abc import abstractmethod
from typing import Union, Optional, List, Set, Tuple

import dgl
import networkx as nx
from grid2op.Action import BaseAction
from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct
from grid2op.Environment import BaseEnv
import torch as th
from grid2op.Observation import BaseObservation
from torch.multiprocessing import cpu_count
import json

from random import choice
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from community_detection.community_detector import CommunityDetector
from multiagent_system.space_factorization import (
    factor_action_space,
    HashableAction,
    factor_observation,
)
from node_agents.utilities import from_networkx_to_dgl, add_self_loop
from utilities import format_to_md
import matplotlib.pyplot as plt


class BasePOP(AgentWithConverter):
    def __init__(
        self,
        env: BaseEnv,
        name: str,
        architecture: Union[str, dict],
        training: bool,
        tensorboard_dir: Optional[str],
        checkpoint_dir: Optional[str],
        seed: int,
        manager_loss: str = "mean",
        device: Optional[str] = None,
        n_jobs: Optional[int] = None,
    ):
        super(BasePOP, self).__init__(env.action_space, IdToAct)

        # Converter
        self.converter = IdToAct(env.action_space)
        self.converter.init_converter()
        self.converter.seed(seed)

        # Reproducibility
        self.seed = seed
        self.name = name
        if device is None:
            self.device: th.device = th.device(
                "cuda:0" if th.cuda.is_available() else "cpu"
            )
        else:
            self.device: th.device = th.device(device)
        if n_jobs is None:
            self.n_jobs = cpu_count() - 1
        else:
            self.n_jobs = n_jobs

        self.architecture: dict = (
            json.load(open(architecture)) if type(architecture) is str else architecture
        )

        # Environment Properties
        self.env = env
        graph = env.reset().as_networkx()
        self.node_features = len(graph.nodes[choice(list(graph.nodes))].keys())
        self.edge_features = len(graph.edges[choice(list(graph.edges))].keys())
        self.training = training
        eb_sched = self.architecture.get("epsilon_beta_scheduling")
        self.epsilon_beta_scheduling = eb_sched if not eb_sched is None else False
        self.manager_loss = manager_loss

        # Checkpointing
        self.checkpoint_dir = checkpoint_dir

        # Logging
        self.trainsteps: int = 0
        self.episodes: int = 0
        self.alive_steps: int = 0
        self.managers_learning_steps: int = 0
        self.agent_learning_steps: int = 0
        self.current_chosen_node: int = -1
        self.current_chosen_manager: int = -1

        # Agents Initialization
        self.action_spaces, self.action_lookup_table = factor_action_space(env)

        self.agent_chosen_actions: List[List[int]] = [
            [] for _ in range(len(self.action_spaces))
        ]

        self.agent_converters: List[IdToAct] = []
        for action_space in self.action_spaces:
            conv = IdToAct(env.action_space)
            conv.init_converter(action_space)
            conv.seed(seed)
            self.agent_converters.append(conv)

        self.local_actions = []
        self.factored_observation = []

        if training:
            # Logging
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=False)
            self.checkpoint_file: str = str(Path(checkpoint_dir, name + ".pt"))

            self.tensorboard_dir = tensorboard_dir
            Path(tensorboard_dir).mkdir(parents=True, exist_ok=False)
            self.writer: Optional[SummaryWriter]
            if tensorboard_dir is not None:
                self.writer = SummaryWriter(log_dir=tensorboard_dir)
                to_log = ""
                for idx, agent_converter in enumerate(self.agent_converters):
                    to_log = (
                        to_log
                        + "Agent "
                        + str(idx)
                        + " has "
                        + str(len(agent_converter.all_actions))
                        + " actions\n"
                    )
                    self.writer.add_text(
                        "Action Spaces/train",
                        format_to_md(to_log),
                        self.trainsteps,
                    )
            else:
                self.writer = None

        # Community Detector Initialization
        self.community_detector = CommunityDetector(seed)
        self.fixed_communities = self.architecture["fixed_communities"]
        self.communities: List[Set[int]] = self.initialize_communities()
        if not self.fixed_communities:
            raise Exception("\nDynamic Communities are not implemented yet\n")

    @property
    @abstractmethod
    def agents(self):
        ...

    @property
    @abstractmethod
    def managers(self):
        ...

    @abstractmethod
    def get_agent_actions(self, factored_observation):
        ...

    @abstractmethod
    def get_manager_actions(self, subgraphs):
        ...

    @abstractmethod
    def get_action(self, *args) -> int:
        ...

    def my_act(
        self,
        transformed_observation: Tuple[List[dgl.DGLHeteroGraph], nx.Graph],
        reward,
        done=False,
    ):

        graph: nx.Graph = transformed_observation[1]

        self.factored_observation: List[dgl.DGLHeteroGraph] = [
            obs for obs in transformed_observation[0]
        ]

        if self.communities and not self.fixed_communities:
            raise Exception("\nDynamic Communities are not implemented yet\n")

        (
            self.local_actions,
            encoded_local_actions,
            current_epsilons,
        ) = self.get_agent_actions(self.factored_observation)

        # Each agent is assigned to its chosen (global) action
        nx.set_node_attributes(
            graph,
            {
                node: {"action": self.lookup_local_action(action), "global_id": node}
                for node, action in zip(graph.nodes, self.local_actions)
            },
        )

        # Splitting into communities
        nx_subgraphs: List[nx.Graph] = []
        zero_edges_nx_subgraphs: List[nx.Graph] = []
        positions = []
        for idx, community in enumerate(self.communities):
            sub_g = graph.subgraph(community)
            if sub_g.number_of_edges == 0:
                positions.append(idx)
                zero_edges_nx_subgraphs.append(sub_g)
            else:
                nx_subgraphs.append(sub_g)

        subgraphs: List[dgl.DGLHeteroGraph] = [
            from_networkx_to_dgl(subgraph, self.device) for subgraph in nx_subgraphs
        ]

        feature_schema = subgraphs[0].edata
        for zero_edge_subg, position in zip(zero_edges_nx_subgraphs, positions):
            subgraphs.insert(
                position, add_self_loop(zero_edge_subg, feature_schema, self.device)
            )

        managed_actions, embedded_graphs, chosen_nodes = self.get_manager_actions(
            subgraphs
        )

        # The graph is summarized by contracting every community in 1 supernode
        summarized_graph = self.summarize_graph(
            graph, embedded_graphs, managed_actions
        ).to(self.device)

        # The head manager chooses the best action from every community
        best_action, self.current_chosen_manager = self.get_action(summarized_graph)

        # Global ID of the current chosen node
        self.current_chosen_node = (
            subgraphs[self.current_chosen_manager]
            .ndata["global_id"][chosen_nodes[self.current_chosen_manager]]
            .item()
        )

        if self.training:
            self.log_to_tensorboard(
                encoded_local_actions,
                best_action,
                managed_actions,
                chosen_nodes,
                current_epsilons,
            )

        return best_action

    def _write_histogram_to_tensorboard(self, to_plot: list, tag: str):
        fig = plt.figure()
        histogram = plt.hist(to_plot, edgecolor="black", linewidth=2)
        plt.xticks((list(set(to_plot))))
        self.writer.add_figure(tag, fig, self.trainsteps)
        plt.close(fig)

    def log_to_tensorboard(
        self,
        encoded_agent_actions,
        best_action,
        managed_actions,
        chosen_nodes,
        current_epsilons,
    ):
        self.writer.add_scalar(
            "Manager Action/Head Manager", best_action, self.trainsteps
        )

        self.writer.add_text(
            "Manager Action/Head Manager",
            format_to_md(str(self.converter.all_actions[best_action])),
            self.trainsteps,
        )
        for idx, obs in enumerate(self.factored_observation):
            self.writer.add_text(
                "Agent Factored Observation/Agent " + str(idx),
                format_to_md(str(obs)),
                self.trainsteps,
            )

        for (idx, action), converter, current_epsilons in zip(
            enumerate(encoded_agent_actions), self.agent_converters, current_epsilons
        ):
            self.writer.add_scalar(
                "Agent Action/Agent " + str(idx), action, self.trainsteps
            )

            self.writer.add_text(
                "Agent Action/Agent " + str(idx),
                format_to_md(str(converter.all_actions[action])),
                self.trainsteps,
            )
            self.writer.add_scalar(
                "Agent Epsilon/Agent " + str(idx),
                current_epsilons,
                self.trainsteps,
            )

        for (idx, action), node in zip(enumerate(managed_actions), chosen_nodes):
            self.writer.add_scalar(
                "Manager Action/Manager " + str(idx),
                action,
                self.trainsteps,
            )
            self.writer.add_text(
                "Manager Action/Manager " + str(idx),
                format_to_md(str(action)),
                self.trainsteps,
            )
            self.writer.add_scalar(
                "Chosen Node/Manager " + str(idx),
                node,
                self.trainsteps,
            )

        self.writer.add_scalar(
            "Chosen Manager/Head Manager", self.current_chosen_manager, self.trainsteps
        )
        self.writer.add_scalar(
            "Chosen Node/Head Manager", self.current_chosen_node, self.trainsteps
        )

    @abstractmethod
    def teach_managers(self, manager_losses):
        ...

    @abstractmethod
    def learn(self, reward: float):
        ...

    @abstractmethod
    def step_agents(self, next_observation, reward, done):
        ...

    def step(
        self,
        reward: float,
        next_observation: BaseObservation,
        done: bool,
    ):
        if done:
            self.writer.add_scalar(
                "POP/Steps Alive per Episode", self.alive_steps, self.episodes
            )
            self.episodes += 1
            self.alive_steps = 0
            self.trainsteps += 1
        else:

            (
                losses,
                q_network_states,
                target_network_states,
                have_learnt,
            ) = self.step_agents(next_observation, reward, done)
            if not (None in losses):
                for idx, loss in enumerate(losses):
                    self.writer.add_scalar(
                        "Agent Loss/Agent " + str(idx), loss, self.agent_learning_steps
                    )
                self.agent_learning_steps += 1
                for (idx, q_state), t_state in zip(
                    enumerate(q_network_states), target_network_states
                ):
                    if q_state is not None and t_state is not None:
                        for (qk, qv), (tk, tv) in zip(q_state.items(), t_state.items()):
                            if qv is not None:
                                self.writer.add_histogram(
                                    "Agent Q Network " + str(qk) + "/Agent " + str(idx),
                                    qv,
                                    self.agent_learning_steps,
                                )
                            if tv is not None:
                                self.writer.add_histogram(
                                    "Agent Target Network "
                                    + str(tk)
                                    + "/Agent "
                                    + str(idx),
                                    tv,
                                    self.agent_learning_steps,
                                )

            self.trainsteps += 1
            self.alive_steps += 1
            self.learn(reward)

    @abstractmethod
    def save(self):
        ...

    @staticmethod
    @abstractmethod
    def load(
        checkpoint_file: str,
        training: bool,
        device: str,
        tensorboard_dir: Optional[str],
        checkpoint_dir: Optional[str],
    ):
        ...

    @staticmethod
    def get_graph(observation: BaseObservation) -> nx.Graph:
        return observation.as_networkx()

    def convert_obs(
        self, observation: BaseObservation
    ) -> Tuple[List[dgl.DGLHeteroGraph], nx.Graph]:
        return factor_observation(observation, self.device, self.architecture["radius"])

    def initialize_communities(self) -> List[Set[int]]:
        obs = self.env.reset()
        graph = self.get_graph(obs)
        communities = self.community_detector.dynamo(graph_t=graph)
        return communities

    def lookup_local_action(self, action: BaseAction):
        return self.action_lookup_table[HashableAction(action)]

    def summarize_graph(
        self,
        graph: nx.Graph,
        embedded_graphs: List[dgl.DGLHeteroGraph],
        managed_actions: List[int],
    ) -> dgl.DGLHeteroGraph:

        # CommunityManager node embedding is assigned to each node
        node_attribute_dict = {}
        for subgraph, community in zip(embedded_graphs, self.communities):
            for idx, node in enumerate(sorted(list(community))):
                node_attribute_dict[node] = {
                    "embedding": subgraph.nodes[idx].data["embedding"].detach()
                }
        nx.set_node_attributes(
            graph,
            node_attribute_dict,
        )

        # Graph is summarized by contracting communities into supernodes
        summarized_graph: nx.graph = graph
        for community in self.communities:
            community_list = list(community)
            for node in community_list[1:]:
                summarized_graph = nx.contracted_nodes(
                    summarized_graph, community_list[0], node
                )

        # Embedding get contracted
        # Each supernode holds the mean of the embedding of the underlying nodes
        for node, node_data in summarized_graph.nodes.data():
            node_data["embedding"] = th.mean(
                th.stack(
                    [
                        node_data["contraction"][contracted_node]["embedding"]
                        for contracted_node in list(node_data["contraction"].keys())
                    ]
                ),
                dim=0,
            ).squeeze()
            for idx, community in enumerate(self.communities):
                if node in community:
                    node_data["action"] = th.Tensor(
                        ((node_data["embedding"].shape[0] - 1) * [0])
                        + [managed_actions[idx]]
                    )
        # The summarized graph is returned in DGL format
        # Each supernode has the action chosen by its community manager
        # And the contracted embedding
        return dgl.from_networkx(
            summarized_graph.to_directed(),
            node_attrs=["action", "embedding"],
            device=self.device,
        )


def train(env: BaseEnv, iterations: int, dpop):

    training_step: int = 0
    obs: BaseObservation = (
        env.reset()
    )  # Typing issue for env.reset(), returns BaseObservation
    done = False
    reward = env.reward_range[0]
    total_episodes = len(env.chronics_handler.subpaths)
    with tqdm(total=iterations - training_step) as pbar:
        while training_step < iterations:
            if dpop.episodes % total_episodes == 0:
                env.chronics_handler.shuffle()
            if done:
                obs = env.reset()
            encoded_action = dpop.my_act(dpop.convert_obs(obs), reward, done)
            action = dpop.convert_act(encoded_action)
            next_obs, reward, done, _ = env.step(action)
            dpop.step(
                reward=reward,
                next_observation=next_obs,
                done=done,
            )
            obs = next_obs
            training_step += 1
            pbar.update(1)

    print("\nSaving...\n")

    dpop.save()
