from torch.multiprocessing import Queue, Process
from typing import Union

from multiagent_system.task import Task, TaskType
from node_agents.base_gcn_agent import BaseGCNAgent


class SerializableGCNAgent(Process, BaseGCNAgent):
    def __init__(
        self,
        agent_actions: int,
        node_features: int,
        edge_features: int,
        architecture: Union[str, dict],
        name: str,
        training: bool,
        device: str,
        task_queue: Queue,
        result_queue: Queue,
    ):
        Process.__init__(self)

        BaseGCNAgent.__init__(
            self,
            agent_actions=agent_actions,
            node_features=node_features,
            edge_features=edge_features,
            architecture=architecture,
            name=name,
            training=training,
            device=device,
        )

        # Multiprocessing
        self.task_queue: Queue = task_queue
        self.result_queue: Queue = result_queue

        # Logging
        self.losses = []
        self.actions_taken = []

    def run(self):
        while True:
            task: Task = self.task_queue.get()
            if task.task_type == TaskType.POISON:
                print("Agent: " + self.name + " shutting down")
                break
            elif task.task_type == TaskType.ACT:
                action = self.take_action(**task.kwargs)
                self.actions_taken.append(action)
                result = action

            elif task.task_type == TaskType.STEP:
                loss = self.step(**task.kwargs)
                self.losses.append(loss)
                result = loss
            elif task.task_type == TaskType.SAVE:
                result = self.save()
            elif task.task_type == TaskType.LOAD:
                self.load(**task.kwargs)
                result = None
            else:
                raise Exception(
                    "Could not recognise task_type: "
                    + str(task.task_type)
                    + " at agent: "
                    + str(self.name)
                )
            self.result_queue.put((self.name, result))

    def save(self):
        return {
            self.name: {
                "optimizer_state": self.optimizer.state_dict(),
                "q_network_state": self.q_network.state_dict(),
                "target_network_state": self.target_network.state_dict(),
                "losses": self.losses,
                "actions": self.actions_taken,
            }
        }

    def load(
        self, optimizer_state, q_network_state, target_network_state, losses, actions
    ):
        self.optimizer.load_state_dict(optimizer_state)
        self.q_network.load_state_dict(q_network_state)
        self.target_network.load_state_dict(target_network_state)
        self.losses = losses
        self.actions_taken = actions
