from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import networkx as nx
from grid2op.Converter import IdToAct
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from pop.multiagent_system.fixed_set import FixedSet


class LoggableModule:
    def __init__(self, tensorboard_dir: Optional[str] = None):
        self.tensorboard_dir: Optional[str] = tensorboard_dir
        self.writer: Optional[SummaryWriter] = None
        if tensorboard_dir is not None:
            if not Path(tensorboard_dir).exists():
                Path(tensorboard_dir).mkdir(parents=True, exist_ok=False)
            self.writer = SummaryWriter(log_dir=tensorboard_dir)

    def is_logging_active(self) -> bool:
        return self.tensorboard_dir is not None

    def log_simple_scalar(
        self, losses: Dict[str, float], train_steps: int, scalar_name: str = "Loss"
    ):
        if self.is_logging_active():
            for iid, loss in losses.items():
                self.writer.add_scalar(scalar_name + "/" + iid, loss, train_steps)

    def log_step(
        self,
        losses: List[Optional[float]],
        implicit_rewards: List[float],
        names: List[str],
        train_steps: int,
    ):
        # Log losses to tensorboard
        self.log_simple_scalar(
            {
                "_".join(agent_name.split("_")[0:2]): loss
                for agent_name, loss in zip(
                    names,
                    losses,
                )
                if loss is not None
            },
            train_steps,
            "Loss",
        )

        self.log_simple_scalar(
            {
                "_".join(agent_name.split("_")[0:2]): implicit_reward
                for agent_name, implicit_reward in zip(names, implicit_rewards)
            },
            train_steps,
            "Implicit Reward",
        )

    def log_action_space_size(self, agent_converters: Dict[int, IdToAct]) -> None:
        if not self.is_logging_active():
            return

        to_log = ""
        for sub_id, agent_converter in agent_converters.items():
            to_log = (
                to_log
                + "Agent "
                + str(sub_id)
                + " has "
                + str(len(agent_converter.all_actions))
                + " actions\n"
            )
            self.writer.add_text(
                "Action Space/Agent " + str(sub_id),
                self._format_to_md(to_log),
                0,
            )

    def log_alive_steps(self, alive_steps: int, episodes: int):
        if self.is_logging_active():
            self.writer.add_scalar("POP/Steps Alive per Episode", alive_steps, episodes)

    def log_agents_loss(self, losses: List[float], agent_learning_steps: int):
        if not self.is_logging_active():
            return

        for idx, loss in enumerate(losses):
            self.writer.add_scalar(
                "Agent Loss/Agent " + str(idx), loss, agent_learning_steps
            )

    def log_communities(self, communities: List[FixedSet], train_steps: int):
        if not self.is_logging_active():
            return
        self.writer.add_text("Communities/POP", str(communities), train_steps)

    def log_graph(self, graph: nx.Graph, train_steps: int):
        if not self.is_logging_active():
            return
        self.writer.add_text(
            "Graph/POP",
            "Graph Nodes: " + str(graph.nodes) + "\nGraph Edges: " + str(graph.edges),
            train_steps,
        )

    def log_agents_embedding_histograms(
        self,
        q_network_states: Tensor,
        target_network_states: Tensor,
        agent_learning_steps: int,
    ):
        if not self.is_logging_active():
            return

        for (idx, q_state), t_state in zip(
            enumerate(q_network_states), target_network_states
        ):
            if q_state is not None and t_state is not None:
                for (qk, qv), (tk, tv) in zip(q_state.items(), t_state.items()):
                    if qv is not None:
                        self.writer.add_histogram(
                            "Agent Q Network " + str(qk) + "/Agent " + str(idx),
                            qv,
                            agent_learning_steps,
                        )
                    if tv is not None:
                        self.writer.add_histogram(
                            "Agent Target Network " + str(tk) + "/Agent " + str(idx),
                            tv,
                            agent_learning_steps,
                        )

    def log_system_behaviour(
        self,
        best_action: int,
        best_action_str: str,
        head_manager_action: int,
        manager_actions: Dict[FrozenSet, Tuple[Set, str]],
        agent_actions: Dict[str, int],
        manager_explorations: Dict[str, Dict[str, Any]],
        agent_explorations: Dict[str, Dict[str, Any]],
        train_steps: int,
    ) -> None:
        if not self.is_logging_active():
            return

        self.writer.add_scalar("POP/Action", best_action, train_steps)

        self.log_head_manager_behaviour(
            best_action=head_manager_action,
            best_action_str=best_action_str,
            train_steps=train_steps,
        )

        self.log_managers_behaviour(
            actions_communities=manager_actions, train_steps=train_steps
        )

        self.log_agents_behaviour(
            agent_actions=agent_actions,
            train_steps=train_steps,
        )

        self.log_multiple_explorations(manager_explorations, train_steps)
        self.log_multiple_explorations(agent_explorations, train_steps)

    def log_multiple_explorations(
        self, exploration_states: Dict[str, Dict[str, Any]], train_steps: int
    ):
        for name, exploration_state in exploration_states.items():
            self.log_exploration(name, exploration_state, train_steps)

    def log_exploration(
        self, name: str, exploration_state: Dict[str, Any], train_steps: int
    ):
        if not self.is_logging_active():
            return
        for section, value in exploration_state.items():
            self.writer.add_scalar(section + "/" + name, value, train_steps)

    def _write_histogram_to_tensorboard(
        self, to_plot: list, tag: str, train_steps: int
    ):
        fig = plt.figure()
        histogram = plt.hist(to_plot, edgecolor="black", linewidth=2)
        plt.xticks((list(set(to_plot))))
        self.writer.add_figure(tag, fig, train_steps)
        plt.close(fig)

    def log_head_manager_behaviour(
        self,
        best_action: int,
        best_action_str: str,
        train_steps: int,
    ) -> None:
        if not self.is_logging_active():
            return
        self.writer.add_scalar("Manager Action/head_manager", best_action, train_steps)
        self.writer.add_text(
            "Manager Action/Head Manager", best_action_str, train_steps
        )

    def log_managers_behaviour(
        self, actions_communities: Dict[Set[int], Tuple[int, str]], train_steps: int
    ):
        community_manager_str = ""
        for community, (action, manager_name) in actions_communities.items():
            self.writer.add_scalar(
                "Manager Action/" + str(manager_name),
                action,
                train_steps,
            )
            community_manager_str += (
                "Community: "
                + str(community)
                + " is managed by "
                + str(manager_name)
                + "\n"
            )
        self.writer.add_text(
            "POP/Manager-Community",
            self._format_to_md(community_manager_str),
            train_steps,
        )

    def log_agents_behaviour(
        self,
        agent_actions: Dict[int, int],
        train_steps: int,
    ):

        for idx, action in agent_actions.items():
            self.writer.add_scalar("Agent Action/" + str(idx), action, train_steps)

    def log_reward(self, reward: float, train_steps: int):
        if self.is_logging_active():
            self.writer.add_scalar("POP/Reward", reward, train_steps)

    def log_penalty(self, penalty: float, train_steps: int):
        if self.is_logging_active():
            self.writer.add_scalar("POP/Repeated Action Penalty", penalty, train_steps)

    @staticmethod
    def _format_to_md(s: str) -> str:
        lines = s.split("\n")
        return "    " + "\n    ".join(lines)
