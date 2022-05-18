from typing import Union, List

from managers.community_manager import CommunityManager
from torch.multiprocessing import Process, Queue

from multiagent_system.task import Task, TaskType
import torch as th


class AsyncCommunityManager(CommunityManager, Process):
    def __init__(
        self,
        node_features: int,
        edge_features: int,
        architecture: Union[str, dict],
        name: str,
        task_queue: Queue,
        result_queue: Queue,
    ):
        Process.__init__(self)
        CommunityManager.__init__(
            self,
            node_features=node_features,
            edge_features=edge_features,
            architecture=architecture,
            name=name,
            log_dir=None,
        )

        # Multiprocessing
        self.task_queue: Queue = task_queue
        self.result_queue: Queue = result_queue

        # Logging
        self.chosen_actions: List[int] = []
        self.losses: List[float] = []

        # Optimizer
        self.optimizer = th.optim.Adam(
            self.parameters(), lr=self.architecture["learning_rate"]
        )

    def run(self):
        while True:
            task: Task = self.task_queue.get()
            if task.task_type == TaskType.CHOOSE_ACTION:
                result = self.forward(**task.kwargs)
                self.chosen_actions.append(result[0])
            elif task.task_type == TaskType.LEARN:
                self.learn(**task.kwargs)
                result = None
            elif task.task_type == TaskType.SAVE:
                result = self.save()
            elif task.task_type == TaskType.LOAD:
                self.async_load(**task.kwargs)
                result = None
            else:
                raise Exception(
                    "Could not recognise task_type: "
                    + str(task.task_type)
                    + " at agent: "
                    + str(self.name)
                )
            self.result_queue.put((self.name, result))

    def learn(self, loss: float):
        loss = th.Tensor(loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.data)

    def save(self) -> dict:
        return {
            self.name: {
                "state": self.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "chosen_actions": self.chosen_actions,
                "losses": self.losses,
            }
        }

    def async_load(self, state_dict, optimizer_state_dict, actions, losses):
        self.load_state_dict(state_dict)
        self.optimizer.load_state_dict(optimizer_state_dict)
        self.chosen_actions = actions
        self.losses = losses
