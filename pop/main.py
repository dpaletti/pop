from pathlib import Path
from grid2op.Agent import BaseAgent
from grid2op.Converter.IdToAct import IdToAct
from grid2op.Reward import CombinedScaledReward, RedispReward, FlatReward
import grid2op
from grid2op.Chronics import MultifolderWithCache
from typing import Union
import grid2op.Environment
import os
from grid2op.Runner import Runner
from agent.gcn_agent import DoubleDuelingGCNAgent, train
from GNN.dueling_gcn import DuelingGCN
from grid2op.Episode import EpisodeData
import shutil
import torch as th
import random
import numpy as np
import dgl


def combine_rewards(env):
    combined_reward: CombinedScaledReward = env.get_reward_instance()
    combined_reward.addReward("Flat", FlatReward(), 0.7)
    combined_reward.addReward("Redispatching", RedispReward(), 0.7)
    # Re-enable this if alarm is supported by the environment
    # combined_reward.addReward("Alarm", AlarmReward(), 0.3)
    combined_reward.initialize(env)


def strip_1_step_episodes(path: str):
    # Copy directory in order not to lose information
    evaluation_path = Path(path)
    shutil.copytree(evaluation_path, Path(path + "_full"))

    episodes = EpisodeData.list_episode(path)
    removed_episodes = 0
    for episode_path, episode_number in episodes:
        episode = EpisodeData.from_disk(episode_path, episode_number)
        if len(episode.actions) == 1:
            removed_episodes += 1
            print("Removing 1-step episode: " + str(episode_number))
            shutil.rmtree(Path(episode_path, episode_number))
    print("Removed " + str(removed_episodes) + " episodes")


def _evaluate(
    env: Union[
        grid2op.Environment.Environment, grid2op.Environment.MultiMixEnvironment
    ],
    agent: BaseAgent,
    path_save: str,
    nb_episode: int = 1,
    nb_process: int = 4,
    sequential: bool = False,
):
    Path(path_save).mkdir(parents=True, exist_ok=True)
    if sequential:
        os.environ[Runner.FORCE_SEQUENTIAL] = "1"
        nb_process = 1
    params = env.get_params_for_runner()
    params["verbose"] = True
    runner = Runner(**params, agentInstance=agent, agentClass=None)
    runner.run(
        nb_episode=nb_episode, nb_process=nb_process, path_save=path_save, pbar=True
    )


def init_train_ddgcn(env):
    name = "ddgcn_conv_medium_seed3"
    q_network = DuelingGCN(
        node_features=4,
        edge_features=11,
        action_space_size=201,
        hidden_size=256,
        hidden_output_size=512,
        value_stream_size=1024,
        advantage_stream_size=1024,
        name="q_network",
        log_dir="/home/l2rpn/data/models/rte/1e4_steps/" + name,
        embedding="conv",
    )
    target_network = DuelingGCN(
        node_features=4,
        edge_features=11,
        action_space_size=201,
        hidden_size=256,
        hidden_output_size=512,
        value_stream_size=1024,
        advantage_stream_size=1024,
        name="target_network",
        log_dir="/home/l2rpn/data/models/rte/1e4_steps/" + name,
        embedding="conv",
    )
    agent = DoubleDuelingGCNAgent(
        action_space=env.action_space,
        q_network=q_network,
        target_network=target_network,
        action_space_converter=IdToAct(env.action_space),
        training=True,
        name="agent",
        log_dir="/home/l2rpn/data/models/rte/1e4_steps/" + name,
        tensorboard_log_dir="/home/l2rpn/data/training/rte/1e4_steps/" + name,
        device="cpu",
    )
    return q_network, target_network, agent


def init_eval_ddgcn(env):
    q_network = DuelingGCN(
        node_features=4,
        edge_features=11,
        action_space_size=201,
        hidden_output_size=256,
        value_stream_size=512,
        advantage_stream_size=512,
        hidden_node_feat_size=[32, 64, 128],
        hidden_edge_feat_size=[32, 64, 128],
        heads=[2, 4, 8],
        embedding="attention",
    )
    target_network = DuelingGCN(
        node_features=4,
        edge_features=11,
        action_space_size=201,
        hidden_output_size=256,
        value_stream_size=512,
        advantage_stream_size=512,
        hidden_node_feat_size=[32, 64, 128],
        hidden_edge_feat_size=[32, 64, 128],
        heads=[2, 4, 8],
        embedding="attention",
    )

    q_network.load("/home/l2rpn/data/models/rte/1e4_steps/ddgcn_conv_egat/q_network.pt")
    target_network.load(
        "/home/l2rpn/data/models/rte/1e4_steps/ddgcn_conv_egat/target_network.pt"
    )

    agent = DoubleDuelingGCNAgent(
        action_space=env.action_space,
        q_network=q_network,
        target_network=target_network,
        action_space_converter=IdToAct(env.action_space),
        training=False,
        device="cpu",
    )
    agent.load("/home/l2rpn/data/models/rte/1e4_steps/ddgcn_egat/agent.pt")
    return q_network, target_network, agent


def fix_seed(seed: int = 0):
    th.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    th.use_deterministic_algorithms(True)
    th.cuda.manual_seed_all(seed)
    dgl.seed(seed)


def main(**kwargs):
    seed = 3
    fix_seed(seed)
    print("Running with seed: " + str(seed))
    # nm_env = "l2rpn_icaps_2021_small"
    nm_env = "rte_case14_realistic"
    env_train = grid2op.make(
        nm_env + "_train80",
        reward_class=CombinedScaledReward,
        chronics_class=MultifolderWithCache,
    )
    env_val = grid2op.make(
        nm_env + "_val10",
        reward_class=CombinedScaledReward,  # CARE Multifolder bugs the runner
    )
    combine_rewards(env_val)
    combine_rewards(env_train)

    # q_network, target_network, agent = init_eval_ddgcn(env_val)
    # _evaluate(
    # env_val,
    # agent,
    # "/home/l2rpn/data/evaluation/rte/1e4_steps/ddgcn_conv_egat",
    # nb_episode=len(env_val.chronics_handler.subpaths),
    # nb_process=1,
    # sequential=True,
    # )

    # evaluate(
    # env_val,
    # "leap_net",
    # "/home/l2rpn/data/models/rte/1e4_steps/",
    # "/home/l2rpn/data/evaluation/rte/1e4_steps/leap_net",
    # len(env_val.chronics_handler.subpaths),
    # 1,
    # verbose=True,
    # )
    # strip_1_step_episodes("/home/l2rpn/data/evaluation/rte/1e4_steps/ddgcn_conv_filter")
    q_network, target_network, agent = init_train_ddgcn(env_train)
    train(
        env_train,
        int(1e4),
        agent,
    )


if __name__ == "__main__":
    main()
