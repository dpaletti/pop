from pathlib import Path
from typing import List, Optional, Set, Tuple, Union
import json
from random import choice
import torch as th


import dgl
import networkx as nx
from grid2op.Action import BaseAction
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import grid2op
from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct
from grid2op.Environment import BaseEnv
from grid2op.Observation import BaseObservation

from managers.async_community_manager import AsyncCommunityManager
from node_agents.serializable_gcn_agent import SerializableGCNAgent
from pop.multiagent_system.task import Task, TaskType
from pop.managers.head_manager import HeadManager
from pop.multiagent_system.space_factorization import (
    factor_action_space,
    factor_observation,
    HashableAction,
)
from pop.node_agents.utilities import from_networkx_to_dgl
from pop.community_detection.community_detector import CommunityDetector
from torch.multiprocessing import cpu_count, Manager, Queue


# TODO: tracking communities in dynamic graphs
# TODO: https://www.researchgate.net/publication/221273637_Tracking_the_Evolution_of_Communities_in_Dynamic_Social_Networks


class DPOP(AgentWithConverter):
    queue_manager: Manager = Manager()

    def __init__(
        self,
        env: BaseEnv,
        name: str,
        architecture: Union[str, dict],
        training: bool,
        tensorboard_dir: Optional[str],
        checkpoint_dir: Optional[str],
        seed: int,
        device: Optional[str] = None,
        n_jobs: Optional[int] = None,
    ):
        super(DPOP, self).__init__(env.action_space, IdToAct)

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
        self.result_queue: Queue = self.queue_manager.Queue()

        self.agents: List[SerializableGCNAgent] = [
            SerializableGCNAgent(
                agent_actions=len(action_space),
                architecture=self.architecture["agent"],
                node_features=self.node_features,
                edge_features=self.edge_features,
                name="agent_" + str(idx) + "_" + name,
                training=training,
                device=device,
                task_queue=self.queue_manager.Queue(),
                result_queue=self.result_queue,
            )
            for idx, action_space in enumerate(action_spaces)
        ]
        self.agent_converters: List[IdToAct] = []
        for action_space in action_spaces:
            conv = IdToAct(env.action_space)
            conv.init_converter(action_space)
            conv.seed(seed)
            self.agent_converters.append(conv)

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
        self.managers: List[AsyncCommunityManager] = [
            AsyncCommunityManager(
                node_features=self.node_features + 1,  # Node Features + Action
                edge_features=self.edge_features,
                architecture=self.architecture["manager"],
                name="manager_" + str(idx) + "_" + name,
                task_queue=self.queue_manager.Queue(),
                result_queue=self.result_queue,
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

        # self.manager_optimizers: List[th.optim.Optimizer] = [
        #     th.optim.Adam(
        #         manager.parameters(), lr=self.architecture["manager"]["learning_rate"]
        #     )
        #     for manager in self.managers
        # ]

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
                Task(TaskType.ACT, transformed_observation=observation)
            )

        return [
            converter.all_actions[encoded_action]
            for encoded_action, converter in zip(
                self.harvest_agent_results(), self.agent_converters
            )
        ]

    def lookup_local_action(self, action: BaseAction):
        return self.action_lookup_table[HashableAction(action)]

    def harvest_agent_results(self):
        result_list = []
        for _ in self.agents:
            result_list.append(self.result_queue.get())

        result_list = [
            (int(agent.split("_")[1]), result) for (agent, result) in result_list
        ]
        result_list = sorted(result_list, key=lambda tup: tup[0])

        return [result for (_, result) in result_list]

    def harvest_manager_results(self):
        result_list = []
        for _ in self.managers:
            result_list.append(self.result_queue.get())

        result_list = [
            (int(manager.split("_")[1]), result) for (manager, result) in result_list
        ]
        result_list = sorted(result_list, key=lambda tup: tup[0])

        return [result for (_, result) in result_list]

    def get_manager_actions(self, subgraphs: List[dgl.DGLHeteroGraph]):
        for mangager, subgraph in zip(self.managers, subgraphs):
            mangager.task_queue.put(Task(TaskType.CHOOSE_ACTION, g=subgraph))

        return zip(*self.harvest_manager_results())

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

        self.local_actions = self.get_agent_actions(self.factored_observation)

        # Each agent is assigned to its chosen (global) action
        nx.set_node_attributes(
            graph,
            {
                node: self.lookup_local_action(action)
                for node, action in zip(graph.nodes, self.local_actions)
            },
            "action",
        )

        # Graph is split into one subgraph per community
        subgraphs: List[dgl.DGLHeteroGraph] = [
            from_networkx_to_dgl(graph.subgraph(community), self.device)
            for community in self.communities
        ]

        managed_actions, embedded_graphs = self.get_manager_actions(subgraphs)

        # Each Community Manager chooses one action from its community
        # manager_decisions = [
        #     manager(subgraph) for manager, subgraph in zip(self.managers, subgraphs)
        # ]
        # managed_actions = [
        #     int(manager_decision[0]) for manager_decision in manager_decisions
        # ]

        # embedded_graphs = [
        #     manager_decision[1] for manager_decision in manager_decisions
        # ]

        # The graph is summarized by contracting every community in 1 supernode
        summarized_graph = self.summarize_graph(
            graph, embedded_graphs, managed_actions
        ).to(self.device)

        # The head manager chooses the best action from every community
        best_action = self.head_manager(summarized_graph)

        if self.training:
            # Tensorboard Logging
            self.writer.add_scalar("Encoded Action/train", best_action, self.trainsteps)
            self.writer.add_text(
                "Action/train",
                str(self.converter.all_actions[best_action]),
                self.trainsteps,
            )

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

    def teach_managers(self, manager_losses):
        for manager, manager_loss in zip(self.managers, manager_losses):
            manager.task_queue.put(Task(TaskType.LEARN, loss=manager_loss))
        self.harvest_manager_results()

    def learn(self, reward: float, losses: List[th.Tensor]):
        if not self.fixed_communities:
            raise Exception("In Learn() we assume fixed communities")
        manager_losses = []
        for community in self.communities:
            community_losses = [
                agent_loss
                for idx, agent_loss in enumerate(losses)
                if idx in community  # !! HERE we are assuming fixed communities
            ]
            loss = sum(community_losses) / len(community_losses)

            manager_losses.append(loss)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

        self.head_manager_optimizer.zero_grad()
        head_manager_loss = th.mean(th.stack(manager_losses))
        head_manager_loss.backward()
        self.head_manager_optimizer.zero_grad()

        # Summary Writer is supposed to not slow down training
        self.save_to_tensorboard(head_manager_loss.mean().item(), reward)

    def step_agents(
        self, next_observation, reward, done
    ) -> Tuple[List[th.Tensor], List[bool]]:
        for (agent, agent_action, agent_observation, agent_next_observation, _,) in zip(
            self.agents,
            self.local_actions,
            self.factored_observation,
            *factor_observation(next_observation, self.device),
        ):
            agent.task_queue.put(
                Task(
                    TaskType.STEP,
                    observation=agent_observation,
                    action=agent_action,
                    reward=reward,
                    next_observation=agent_next_observation,
                    done=done,
                )
            )

        losses = self.harvest_agent_results()

        return losses, list(map(lambda x: not (x is None), losses))

    def step(
        self,
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
            losses, have_learnt = self.step_agents(next_observation, reward, done)
            self.trainsteps += 1
            self.alive_steps += 1
            if all(have_learnt):
                self.learn(reward, losses)
                self.learning_steps += 1

    def save(self):
        for agent in self.agents:
            agent.task_queue.put(Task(TaskType.SAVE))
        agents_state = dict(self.harvest_agent_results())

        for manager in self.managers:
            manager.task_queue.put(Task(TaskType.SAVE))
        managers_state = dict(self.harvest_manager_results())

        checkpoint = {
            "agents_state": agents_state,
            "managers_state": managers_state,
            "head_manager_state": self.head_manager.state_dict(),
            "head_manager_optimizer_state": self.head_manager_optimizer.state_dict(),
            "trainsteps": self.trainsteps,
            "episodes": self.episodes,
            "name": self.name,
            "env_name": self.env.env_name,
            "architecture": self.architecture,
            "seed": self.seed,
        }
        th.save(checkpoint, self.checkpoint_dir)

    def save_to_tensorboard(self, loss: float, reward: float) -> None:
        self.writer.add_scalar("Loss/train", loss, self.trainsteps)
        self.writer.add_scalar("Reward/train", reward, self.trainsteps)

    @staticmethod
    def load(
        checkpoint_file: str,
        training: bool,
        device: str,
        tensorboard_dir: Optional[str] = None,
    ):
        checkpoint = th.load(checkpoint_file)

        dpop = DPOP(
            env=grid2op.make(checkpoint["env_name"]),
            architecture=checkpoint["architecture"],
            name=checkpoint["name"],
            training=training,
            tensorboard_dir=tensorboard_dir,
            checkpoint_dir=Path(checkpoint_file).parents[0],
            seed=checkpoint["seed"],
            device=device,
        )

        dpop.trainsteps = checkpoint["trainsteps"]
        dpop.episodes = checkpoint["episodes"]

        for idx, agent in enumerate(dpop.agents):
            agent.task_queue.put(
                Task(
                    TaskType.LOAD,
                    optimizer_state=checkpoint["agents_state"][
                        "agent_" + str(idx) + "_" + dpop.name
                    ]["optimizer_state"],
                    q_network_state=checkpoint["agents_state"][
                        "agent_" + str(idx) + "_" + dpop.name
                    ]["q_network_state"],
                    target_network_state=checkpoint["agents_state"][
                        "agent_" + str(idx) + "_" + dpop.name
                    ]["target_network_state"],
                    losses=checkpoint["agents_state"][
                        "agent_" + str(idx) + "_" + dpop.name
                    ]["losses"],
                    actions=checkpoint["agents_state"][
                        "agent_" + str(idx) + "_" + dpop.name
                    ]["actions"],
                )
            )
        dpop.harvest_agent_results()

        for idx, manager in enumerate(dpop.managers):
            manager.task_queue.put(
                Task(
                    TaskType.LOAD,
                    state_dict=checkpoint["managers_state"][
                        "manager_" + str(idx) + "_" + dpop.name
                    ]["state"],
                    optimizer_state_dict=checkpoint["managers_state"][
                        "manager_" + str(idx) + "_" + dpop.name
                    ]["optimizer_state"],
                    action=checkpoint["managers_state"][
                        "manager_" + str(idx) + "_" + dpop.name
                    ]["chosen_actions"],
                    losses=checkpoint["managers_state"][
                        "manager_" + str(idx) + "_" + dpop.name
                    ]["losses"],
                )
            )
        dpop.harvest_manager_results()


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
                reward=reward,
                next_observation=next_obs,
                done=done,
            )
            obs = next_obs
            training_step += 1
            pbar.update(1)

    print("\nSaving...\n")

    dpop.save()
