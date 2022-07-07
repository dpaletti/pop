from pathlib import Path
from typing import Optional, List

from grid2op.Converter import IdToAct
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter


class LoggableModule:
    def __init__(self, tensorboard_dir: Optional[str] = None):
        self.tensorboard_dir: Optional[str] = tensorboard_dir
        self.writer: Optional[SummaryWriter] = None
        if tensorboard_dir is not None:
            Path(tensorboard_dir).mkdir(parents=True, exist_ok=False)
            self.writer = SummaryWriter(log_dir=tensorboard_dir)

    def is_logging_active(self) -> bool:
        return self.tensorboard_dir is not None

    def log_action_space_size(self, agent_converters: List[IdToAct]) -> None:
        if not self.is_logging_active():
            return

        to_log = ""
        for idx, agent_converter in enumerate(agent_converters):
            to_log = (
                to_log
                + "Agent "
                + str(idx)
                + " has "
                + str(len(agent_converter.all_actions))
                + " actions\n"
            )
            self.writer.add_text(
                "Action Space/Agent " + str(idx),
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
        manager_actions: List[int],
        agent_actions: List[int],
        best_node: int,
        manager_nodes: List[int],
        best_manager: int,
        converter: IdToAct,
        agent_converters: List[IdToAct],
        agent_epsilons: List[float],
        train_steps: int,
    ) -> None:
        if not self.is_logging_active():
            return

        self.log_head_manager_behaviour(
            best_action=best_action,
            train_steps=train_steps,
            converter=converter,
            chosen_manager=best_manager,
            chosen_node=best_node,
        )

        self.log_managers_behaviour(
            actions=manager_actions, chosen_nodes=manager_nodes, train_steps=train_steps
        )

        self.log_agents_behaviour(
            converters=agent_converters,
            agent_actions=agent_actions,
            epsilons=agent_epsilons,
            train_steps=train_steps,
        )

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
        train_steps: int,
        converter: IdToAct,
        chosen_manager: int,
        chosen_node: int,
    ) -> None:
        if not self.is_logging_active():
            return
        self.writer.add_scalar("Manager Action/Head Manager", best_action, train_steps)
        self.writer.add_text(
            "Manager Action/Head Manager",
            self._format_to_md(str(converter.all_actions[best_action])),
            train_steps,
        )
        self.writer.add_scalar(
            "Chosen Manager/Head Manager", chosen_manager, train_steps
        )
        self.writer.add_scalar("Chosen Node/Head Manager", chosen_node, train_steps)

    def log_managers_behaviour(
        self, actions: List[int], chosen_nodes: List[int], train_steps: int
    ):

        for (idx, action), node in zip(enumerate(actions), chosen_nodes):
            self.writer.add_scalar(
                "Manager Action/Manager " + str(idx),
                action,
                train_steps,
            )
            self.writer.add_text(
                "Manager Action/Manager " + str(idx),
                self._format_to_md(str(action)),
                train_steps,
            )
            self.writer.add_scalar(
                "Chosen Node/Manager " + str(idx),
                node,
                train_steps,
            )

    def log_agents_behaviour(
        self,
        converters: List[IdToAct],
        agent_actions: List[int],
        epsilons: List[float],
        train_steps: int,
    ):

        for (idx, action), converter, current_epsilons in zip(
            enumerate(agent_actions), converters, epsilons
        ):
            self.writer.add_scalar(
                "Agent Action/Agent " + str(idx), action, train_steps
            )

            self.writer.add_text(
                "Agent Action/Agent " + str(idx),
                self._format_to_md(str(converter.all_actions[action])),
                train_steps,
            )
            self.writer.add_scalar(
                "Agent Epsilon/Agent " + str(idx),
                current_epsilons,
                train_steps,
            )

    @staticmethod
    def _format_to_md(s: str) -> str:
        lines = s.split("\n")
        return "    " + "\n    ".join(lines)
