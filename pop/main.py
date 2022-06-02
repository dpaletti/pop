import toml

from pathlib import Path
from grid2op.Agent import BaseAgent
from grid2op.Environment import BaseEnv
from grid2op.Reward import (
    CombinedScaledReward,
    RedispReward,
    FlatReward,
    AlarmReward,
)
import grid2op
from grid2op.Chronics import MultifolderWithCache
from typing import Union
import grid2op.Environment
import grid2op.Reward as R
from grid2op.Runner import Runner
from grid2op.Episode import EpisodeData
import shutil
import torch as th
import random
import numpy as np
import dgl
import os
import resource

from pop.multiagent_system.ray_dpop import RayDPOP
from pop.multiagent_system.base_pop import train
import argparse
import importlib
import pprint


def set_l2rpn_reward(env, alarm: bool = True):
    combined_reward: CombinedScaledReward = env.get_reward_instance()
    combined_reward.addReward("Flat", FlatReward(), 0.7)
    combined_reward.addReward("Redispatching", RedispReward(), 0.7)
    if alarm:
        combined_reward.addReward("Alarm", AlarmReward(), 0.3)
    else:
        print("\nWARNING: Alarm Reward deactivated\n")
    combined_reward.initialize(env)


def set_reward(env, config):
    combined_reward: CombinedScaledReward = env.get_reward_instance()
    grid2op_reward_module = importlib.import_module("grid2op.Reward")
    for reward_name, reward_weight in config["environment"]["reward"].items():
        combined_reward.addReward(
            reward_name,
            getattr(grid2op_reward_module, reward_name + "Reward")(),
            reward_weight,
        )

    combined_reward.initialize(env)


def set_topological_reward(env, alarm: bool = True):
    combined_reward: CombinedScaledReward = env.get_reward_instance()
    combined_reward.addReward("Bridge", R.BridgeReward(), 0.2)
    combined_reward.addReward("CloseToOverflow", R.CloseToOverflowReward(), 0.2)
    combined_reward.addReward("EpisodeDuration", R.EpisodeDurationReward(), 0.1)
    combined_reward.addReward("Gameplay", R.GameplayReward(), 0.1)
    combined_reward.addReward("LinesCapacity", R.LinesCapacityReward(), 0.1)
    combined_reward.addReward("LinesReconnected", R.LinesReconnectedReward(), 0.2)
    combined_reward.addReward("Redispatching", R.RedispReward(), 0.1)
    if alarm:
        combined_reward.addReward("Alarm", AlarmReward(), 0.3)
    else:
        print("\n WARNING: Alarm Reward deactivated\n")
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
    os.environ["OMP_NUM_THREADS"] = "1"

    # How OMP Threads are scheduled
    # os.environ["OMP_SCHEDULE"] = "STATIC"

    # Whether thread may be moved between processors
    # os.environ["OMP_PROC_BIND"] = "CLOSE"

    # CPU Affinity is not set, you need to call with taskset -c to set cpu affinity


def limit_torch_multiprocessing_resources():
    # If virtual memory runs out
    # ulimit -n 10000 is another solution on top of the sharing_stratecy

    th.multiprocessing.set_sharing_strategy("file_system")
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def _find_field(config: dict, find_field_name: str) -> str:
    for section_name, section_content in config.items():
        for field_name, field_content in section_content.items():
            if find_field_name == field_name:
                return field_content
    raise Exception("Parsing failed: " + str(find_field_name) + "does not exist")


def parse_config(run_file_path: str) -> dict:
    # Parse config file
    config = toml.load(open(run_file_path))
    for section_name, section_content in config.items():
        for field_name, field_content in section_content.items():
            if isinstance(field_content, str):
                open_double_brackets = field_content.find("{{")
                while open_double_brackets > -1:
                    closed_double_brackets = field_content.find("}}")
                    field_content = (
                        field_content[:open_double_brackets]
                        + _find_field(
                            config,
                            field_content[
                                open_double_brackets + 2 : closed_double_brackets
                            ],
                        )
                        + field_content[closed_double_brackets + 2 :]
                    )
                    open_double_brackets = field_content.find("{{")
                section_content[field_name] = field_content
    return config


def main(**kwargs):
    set_environment_variables(disable_gpu=True)

    config = parse_config(kwargs["run_file"])

    # In case of memory failures
    # limit_torch_multiprocessing_resources()

    # Use Processes not Threads
    th.multiprocessing.set_start_method("spawn", force=True)

    # nm_env = "l2rpn_icaps_2021_small"
    nm_env = config["environment"]["name"]
    print("Running with env: " + nm_env)
    # Build train and validation environments
    env_train = grid2op.make(
        nm_env + "_train80",
        reward_class=CombinedScaledReward,
        chronics_class=MultifolderWithCache,
        difficulty="competition",
    )
    env_val = grid2op.make(
        nm_env + "_val10",
        reward_class=CombinedScaledReward,  # CARE Multifolder bugs the runner
    )
    curriculum_envs = []
    if config["training"]["train"] and config["training"]["curriculum"]:
        env_train_0 = grid2op.make(
            nm_env + "_train80",
            reward_class=CombinedScaledReward,
            chronics_class=MultifolderWithCache,
            difficulty=0,
        )

        env_train_1 = grid2op.make(
            nm_env + "_train80",
            reward_class=CombinedScaledReward,
            chronics_class=MultifolderWithCache,
            difficulty=1,
        )

        env_train_2 = grid2op.make(
            nm_env + "_train80",
            reward_class=CombinedScaledReward,
            chronics_class=MultifolderWithCache,
            difficulty=2,
        )

        env_train_competition = grid2op.make(
            nm_env + "_train80",
            reward_class=CombinedScaledReward,
            chronics_class=MultifolderWithCache,
            difficulty="competition",
        )
        curriculum_envs = [env_train_0, env_train_1, env_train_2, env_train_competition]
        # TODO: set different rewards in curriculum learning
        # TODO: do so that at easier difficulties we reward having a very well working powernet
        # TODO: while in later stages we reward surviving the longest
        # TODO: do so by moving the importance of reward factors at each difficulty change
        for env in curriculum_envs:
            set_reward(env, config)

    # Set seed for reproducibility
    seed = config["reproducibility"]["seed"]
    print("Running with seed: " + str(seed))
    fix_seed(env_train, env_val, seed=seed)

    # Set reward
    set_reward(env_train, config)
    set_l2rpn_reward(env_val, alarm=False)

    if config["loading"]["load"]:
        # Load agent
        agent = RayDPOP.load(
            checkpoint_file=config["loading"]["load_dir"],
            training=config["training"]["train"],
            device=config["reproducibility"]["device"],
            tensorboard_dir=config["training"]["tensorboard_dir"],
            checkpoint_dir=config["model"]["checkpoint_dir"],
        )
    else:
        # Instantiate agent ex novo
        agent = RayDPOP(
            env=env_train,
            name=config["model"]["name"],
            architecture=config["model"]["architecture_path"],
            training=config["training"]["train"],
            tensorboard_dir=config["training"]["tensorboard_dir"],
            checkpoint_dir=config["model"]["checkpoint_dir"],
            seed=seed,
            device=config["reproducibility"]["device"],
        )

    try:
        agent.writer.add_text("Description/train", pprint.pformat(config))
        agent.writer.add_text("Architecture/train", pprint.pformat(agent.architecture))
    except:
        pass

    if config["training"]["train"]:
        if not config["training"]["curriculum"]:
            train(
                env=env_train, dpop=agent, iterations=int(config["training"]["steps"])
            )
        else:
            steps = int(config["training"]["steps"] / 4)
            for idx, env in enumerate(curriculum_envs):
                # TODO: reload, set the timestamps correctly
                # TODO: avoid overwriting previous data

                train(env=env, dpop=agent, iterations=steps)
                agent = RayDPOP.load(
                    checkpoint_file=config["loading"]["load_dir"],
                    training=config["training"]["train"],
                    device=config["reproducibility"]["device"],
                    tensorboard_dir=config["training"]["tensorboard_dir"]
                    + "_curr_"
                    + str(idx + 1),
                    checkpoint_dir=config["model"]["checkpoint_dir"]
                    + "_curr_"
                    + str(idx + 1),
                )
    else:
        _evaluate(
            env_val,
            agent,
            config["evaluation"]["evaluation_dir"],
            nb_episode=config["evaluation"]["episodes"],
            sequential=True,
        )


p = argparse.ArgumentParser()
p.add_argument("--run-file", type=str)


if __name__ == "__main__":
    args = p.parse_args()
    main(**vars(args))
