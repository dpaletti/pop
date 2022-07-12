import grid2op
from grid2op.Chronics import MultifolderWithCache
from grid2op.Reward import CombinedScaledReward
from ray.util.client import ray

from agents.manager import Manager
from configs.run_config import RunConfiguration
from main import set_reward


def test_base_pop():
    config = RunConfiguration("../run_files/dpop_rte_1e4.toml")

    # Train Environment
    env_train = grid2op.make(
        config.environment.name + "_train80",
        reward_class=CombinedScaledReward,
        chronics_class=MultifolderWithCache,
        difficulty="competition",
    )
    set_reward(env_train, config)

    manager = Manager.remote(
        agent_actions=14,
        node_features=8 + 1,  # Node Features + Action
        edge_features=6,
        architecture=config.model.architecture.manager,
        name="some",
        training=True,
        device="cpu",
    )
