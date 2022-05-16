from pathlib import Path
from grid2op.Agent import BaseAgent
from grid2op.Environment import BaseEnv
from grid2op.Reward import CombinedScaledReward, RedispReward, FlatReward, AlarmReward
import grid2op
from grid2op.Chronics import MultifolderWithCache
from typing import Union
import grid2op.Environment
from grid2op.Runner import Runner
from grid2op.Episode import EpisodeData
import shutil
import torch as th
import random
import numpy as np
import dgl
import os
import multiprocessing

import dill as pickle  # Needed for multiprocessing
from pop.multiagent_system.DPOP import DPOP, train


def combine_rewards(env, alarm: bool = True):
    combined_reward: CombinedScaledReward = env.get_reward_instance()
    combined_reward.addReward("Flat", FlatReward(), 0.7)
    combined_reward.addReward("Redispatching", RedispReward(), 0.7)
    if alarm:
        combined_reward.addReward("Alarm", AlarmReward(), 0.3)
    else:
        print("\nWARNING: Alarm Reward deactivated\n")
    combined_reward.initialize(env)


def strip_1_step_episodes(path: str):
    # This method is used to avoid grid2viz crashing on 1 step episodes
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


def fix_seed(env_train: BaseEnv, env_val: BaseEnv, seed: int = 0):
    env_train.seed(seed)
    env_val.seed(seed)
    th.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    th.use_deterministic_algorithms(True)
    th.cuda.manual_seed_all(seed)
    dgl.seed(seed)


def set_environment_variables(disable_gpu: bool):
    if disable_gpu:
        # Removing all GPUs from visible devices
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Set the number of threads used by Open MPI (OMP)
    os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())

    # How OMP Threads are scheduled
    os.environ["OMP_SCHEDULE"] = "STATIC"

    # Whether thread may be moved between processors
    os.environ["OMP_PROC_BIND"] = "CLOSE"

    # CPU Affinity is not set, you need to call with taskset -c to set cpu affinity


def main():
    set_environment_variables(disable_gpu=True)

    # nm_env = "l2rpn_icaps_2021_small"
    nm_env = "rte_case14_realistic"

    print("Running with env: " + "rte_case14_realistic")

    env_train = grid2op.make(
        nm_env + "_train80",
        reward_class=CombinedScaledReward,
        chronics_class=MultifolderWithCache,
    )
    env_val = grid2op.make(
        nm_env + "_val10",
        reward_class=CombinedScaledReward,  # CARE Multifolder bugs the runner
    )

    seed = 0
    print("Running with seed: " + str(seed))
    fix_seed(env_train, env_val, seed=seed)

    combine_rewards(env_val, alarm=False)
    combine_rewards(env_train, alarm=False)

    agent = DPOP(
        env=env_train,
        name="dpop_rte_1e4",
        architecture_path="../architectures/dpop_agent_xxs.json",
        training=True,
        tensorboard_dir="../test_data/pop_runs/tensorboard/dpop_rte_1e4",
        checkpoint_dir="../test_data/pop_runs/checkpoint/dpop_rte_1e4",
        seed=0,
        device="cpu",
    )

    train(env=env_train, dpop=agent, iterations=int(1e4))


if __name__ == "__main__":
    main()
