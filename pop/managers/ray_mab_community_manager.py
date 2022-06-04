from typing import Union

import ray
import vowpalwabbit as vw

from graph_convolutional_networks.egat_gcn import EgatGCN
from graph_convolutional_networks.gcn import GCN
from managers.manager import Manager
import torch as th
from pathlib import Path


@ray.remote
class RayMABCommunityManager(Manager):
    def __init__(
        self,
        architecture: Union[str, dict],
        node_features: int,
        edge_features: int,
        name: str,
        log_dir: str,
    ):
        super(RayMABCommunityManager, self).__init__(
            architecture=architecture,
            node_features=node_features,
            edge_features=edge_features,
            name=name,
            log_dir=log_dir,
        )

        self._embedding = EgatGCN(
            node_features,
            edge_features,
            self.architecture["embedding_architecture"],
            name + "_embedding",
            log_dir,
        ).float()

        if log_dir:
            Path(self.log_dir).mkdir(parents=True, exist_ok=False)
            self.mab_file = str(Path(self.log_dir, name + "_mab.model"))
        # TODO: care, VW takes costs not rewards remember to invert it
        # TODO: probably the embedding in this case just returns a all the features available from the graph
        # TODO: remove some neural networks from this hell
        # "--cb_explore_adf --softmax --lambda 10"
        self._mab = vw.Workspace(self.architecture["mab"])

    @property
    def node_choice(self) -> vw.Workspace:
        return self._mab

    @property
    def embedding(self) -> GCN:
        return self._embedding

    def learn(self, loss: float):
        loss = th.Tensor(loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.data)

    def get_state(self):
        self._mab.save(self.log_dir)
        return [
            self.embedding.state_dict(),
            self.optimizer.state_dict(),
            self.chosen_actions,
            self.losses,
        ]

    def get_embedding(self):
        return self.embedding

    def get_name(self):
        return self.name

    def load_state(
        self,
        embedding_state_dict,
        node_attention_state_dict,
        optimizer_state_dict,
        actions,
        losses,
    ):
        self._mab = vw.Workspace(self.architecture["mab"] + " -i " + self.mab_file)
        self.embedding.load_state_dict(embedding_state_dict)
        self.node_choice.state_dict(node_attention_state_dict)
        self.optimizer.load_state_dict(optimizer_state_dict)
        self.chosen_actions = actions
        self.losses = losses
