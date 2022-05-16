from pathlib import Path
from typing import List, Optional, Set, Tuple
import json
from random import choice
import torch as th


import dgl
import networkx as nx
from grid2op.Action import BaseAction
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct
from grid2op.Environment import BaseEnv
from grid2op.Observation import BaseObservation

from multiagent_system.agent_task import AgentTask, TaskType
from pop.managers.head_manager import HeadManager
from pop.multiagent_system.space_factorization import (
    factor_action_space,
    factor_observation,
    HashableAction,
)
from pop.node_agents.utilities import from_networkx_to_dgl
from pop.node_agents.gcn_agent import DoubleDuelingGCNAgent
from pop.community_detection.community_detector import CommunityDetector
from pop.managers.community_manager import CommunityManager
from multiprocessing import cpu_count, JoinableQueue

# TODO: tracking communities in dynamic graphs
# TODO: https://www.researchgate.net/publication/221273637_Tracking_the_Evolution_of_Communities_in_Dynamic_Social_Networks


class DPOP(AgentWithConverter):
    def __init__(
        self,
        env: BaseEnv,
        name: str,
        architecture_path: str,
        training: bool,
        tensorboard_dir: Optional[str],
        checkpoint_dir: Optional[str],
        seed: int,
        device: Optional[str] = None,
        n_jobs: Optional[int] = None,
    ):
        super(DPOP, self).__init__(env.action_space, IdToAct)

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

        self.architecture = json.load(open(architecture_path))

        # Environment Properties
        self.env = env
        graph = env.reset().as_networkx()
        self.node_features = len(graph.nodes[choice(list(graph.nodes))].keys())
        self.edge_features = len(graph.edges[choice(list(graph.edges))].keys())

        # Checkpointing
        self.checkpoint_dir = checkpoint_dir

        # Logging
        self.trainsteps: int = 0
        self.episodes: int = 0
        self.alive_steps: int = 0
        self.learning_steps: int = 0

        self.tensorboard_dir = tensorboard_dir
        if training:
            Path(tensorboard_dir).mkdir(parents=True, exist_ok=True)
            self.writer: Optional[SummaryWriter]
            if tensorboard_dir is not None:
                self.writer = SummaryWriter(log_dir=tensorboard_dir)
            else:
                self.writer = None

        # Agents Initialization
        action_spaces, self.action_lookup_table = factor_action_space(env)

        # Multiprocessing
        self.result_queue: JoinableQueue = JoinableQueue()
        self.agents: List[DoubleDuelingGCNAgent] = [
            DoubleDuelingGCNAgent(
                agent_actions=action_space,
                full_action_space=env.action_space,
                architecture=self.architecture["agent"],
                node_features=self.node_features,
                edge_features=self.edge_features,
                name="agent_" + str(idx) + "_" + name,
                seed=seed,
                training=training,
                tensorboard_log_dir=tensorboard_dir,
                log_dir=self.checkpoint_dir,
                device=device,
                task_queue=JoinableQueue(),
                result_queue=self.result_queue,
            )
            for idx, action_space in enumerate(action_spaces)
        ]

        # Start agent process
        for agent in self.agents:
            agent.start()

        self.local_actions = []
        self.factored_observation = []

        # Community Detector Initialization
        self.community_detector = CommunityDetector(seed)
        self.fixed_communities = self.architecture["fixed_communities"]
        self.communities: List[Set[int]] = self.initialize_communities()
        if not self.fixed_communities:
            raise Exception("\nDynamic Communities are not implemented yet\n")

        # Managers Initialization
        self.managers: List[CommunityManager] = [
            CommunityManager(
                node_features=self.node_features + 1,  # Node Features + Action
                edge_features=self.edge_features,
                architecture=self.architecture["manager"],
                log_dir=self.checkpoint_dir,
                name="manager_" + str(idx) + "_" + name,
            ).to(device)
            for idx, _ in enumerate(self.communities)
        ]

        self.head_manager = HeadManager(
            node_features=self.managers[0].get_embedding_dimension()
            * 2,  # Manager Embedding + Action (padded)
            architecture=self.architecture["head_manager"],
            name="head_manager_" + "_" + name,
            log_dir=self.checkpoint_dir,
        ).to(device)
        self.manager_optimizers: List[th.optim.Optimizer] = [
            th.optim.Adam(
                manager.parameters(), lr=self.architecture["manager"]["learning_rate"]
            )
            for manager in self.managers
        ]

        self.head_manager_optimizer: th.optim.Optimizer = th.optim.Adam(
            self.head_manager.parameters(),
            lr=self.architecture["head_manager"]["learning_rate"],
        )

    def initialize_communities(self) -> List[Set[int]]:
        obs = self.env.reset()
        graph = self.get_graph(obs)
        communities = self.community_detector.dynamo(graph_t=graph)
        return communities

    def get_agent_actions(self, factored_observation):
        for observation, agent in zip(factored_observation, self.agents):
            agent.task_queue.put(
                AgentTask(TaskType.ACT, transformed_observation=observation)
            )
        return self.harvest_results()

    def lookup_local_action(self, action: BaseAction):
        return self.action_lookup_table[HashableAction(action)]

    def harvest_results(self):
        for agent in self.agents:
            agent.task_queue.join()
        result_list = []
        while not self.result_queue.empty():
            result_list.append(self.result_queue.get())
        result_list = [
            (int(agent.split("_")[1]), action) for (agent, action) in result_list
        ]
        result_list = sorted(result_list, key=lambda tup: tup[0])
        print("SORTED LIST" + str(result_list))
        return [result for (_, result) in result_list]

    def my_act(
        self,
        transformed_observation: Tuple[List[dgl.DGLHeteroGraph], nx.Graph],
        reward: float,
        done=False,
    ) -> int:
        graph: nx.Graph = transformed_observation[1]

        self.factored_observation: List[dgl.DGLHeteroGraph] = [
            obs for obs in transformed_observation[0]
        ]

        if self.communities and not self.fixed_communities:
            raise Exception("\nDynamic Communities are not implemented yet\n")

        local_actions = self.get_agent_actions(self.factored_observation)

        # Each agent is assigned to its chosen (global) action
        nx.set_node_attributes(
            graph,
            {
                node: self.lookup_local_action(action)
                for node, action in zip(graph.nodes, local_actions)
            },
            "action",
        )

        # Graph is split into one subgraph per community
        subgraphs: List[dgl.DGLHeteroGraph] = [
            from_networkx_to_dgl(graph.subgraph(community), self.device)
            for community in self.communities
        ]

        # Each Community Manager chooses one action from its community
        manager_decisions = [
            manager(subgraph) for manager, subgraph in zip(self.managers, subgraphs)
        ]
        managed_actions = [
            int(manager_decision[0]) for manager_decision in manager_decisions
        ]

        embedded_graphs = [
            manager_decision[1] for manager_decision in manager_decisions
        ]

        # The graph is summarized by contracting every community in 1 supernode
        summarized_graph = self.summarize_graph(
            graph, embedded_graphs, managed_actions
        ).to(self.device)

        # The head manager chooses the best action from every community
        best_action = self.head_manager(summarized_graph)
        print("CHOSEN BEST ACTION")
        print(best_action)

        return best_action

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

    def convert_obs(
        self, observation: BaseObservation
    ) -> Tuple[List[dgl.DGLHeteroGraph], nx.Graph]:
        return factor_observation(
            observation,
            self.device,
            self.architecture["radius"],
        )

    @staticmethod
    def get_graph(observation: BaseObservation) -> nx.Graph:
        return observation.as_networkx()

    def save_to_tensorboard(self, loss: float, reward: float) -> None:
        if self.writer is None:
            print("Warning: trying to save to tensorboard but it's deactivated")

        self.writer.add_scalar("Loss/train", loss, self.trainsteps)
        self.writer.add_scalar("Reward/train", reward, self.trainsteps)

    def learn(self, reward: float):
        if not self.fixed_communities:
            raise Exception("In Learn() we assume fixed communities")
        manager_losses = []
        for optimizer, community in zip(self.manager_optimizers, self.communities):
            community_losses = [
                agent.loss
                for idx, agent in enumerate(self.agents)
                if idx in community  # !! HERE we are assuming fixed communities
            ]
            loss = sum(community_losses) / len(community_losses)

            manager_losses.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.head_manager_optimizer.zero_grad()
        head_manager_loss = th.mean(th.stack(manager_losses))
        head_manager_loss.backward()
        self.head_manager_optimizer.zero_grad()
        self.save(head_manager_loss.mean().item(), reward)

    def step(
        self,
        observation: BaseObservation,
        action: int,
        reward: float,
        next_observation: BaseObservation,
        done: bool,
    ):
        if done:
            if self.writer is not None:
                self.writer.add_scalar(
                    "Steps Alive per Episode", self.alive_steps, self.episodes
                )
            self.episodes += 1
            self.alive_steps = 0
            self.trainsteps += 1
        else:
            for (
                agent,
                agent_action,
                agent_observation,
                agent_next_observation,
                _,
            ) in zip(
                self.agents,
                self.local_actions,
                self.factored_observation,
                *factor_observation(next_observation, self.device),
            ):
                agent.step(
                    agent_observation,
                    agent_action,
                    reward,
                    agent_next_observation,
                    done,
                )
            self.trainsteps += 1
            self.alive_steps += 1
            if (
                self.trainsteps % self.architecture["agent"]["learning_frequency"]
                and len(self.agents[0].memory)
                < self.architecture["agent"]["batch_size"]
                == 0
            ):
                self.learn(reward)
                self.learning_steps += 1

    def save(self, head_manager_loss: float, reward: float):
        checkpoint = {
            "manager_optimizers_states": [
                optimizer.state_dict() for optimizer in self.manager_optimizers
            ],
            "head_manager_optimizer_states": self.head_manager_optimizer.state_dict(),
            "trainsteps": self.trainsteps,
            "episodes": self.episodes,
            "name": self.name,
            "architecture": self.architecture,
            "tensorboard_dir": self.tensorboard_dir,
            "checkpoint_dir": self.checkpoint_dir,
            "seed": self.seed,
            "node_features": self.node_features,
            "edge_features": self.edge_features,
        }

        for manager in self.managers:
            manager.save()

        self.head_manager.save()

        th.save(checkpoint, self.checkpoint_dir)
        self.save_to_tensorboard(head_manager_loss, reward)


def train(env: BaseEnv, iterations: int, dpop: DPOP):

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
                observation=obs,
                action=encoded_action,
                reward=reward,
                next_observation=next_obs,
                done=done,
            )
            obs = next_obs
            training_step += 1
            pbar.update(1)
