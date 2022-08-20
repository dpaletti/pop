from pathlib import Path
from grid2op.Agent import BaseAgent
from grid2op.Environment import BaseEnv
from grid2op.Reward import (
    CombinedScaledReward,
    RedispReward,
    IncreasingFlatReward,
    AlarmReward,
)
from grid2op.Chronics import MultifolderWithCache
from typing import Union
import grid2op.Environment
from grid2op.Runner import Runner
import torch as th
import random
import numpy as np
import dgl
import os

from configs.run_config import RunConfiguration
import argparse
import importlib

import warnings

from multiagent_system.dpop import DPOP
from pop.multiagent_system.base_pop import train

warnings.filterwarnings("ignore", category=UserWarning)


def set_l2rpn_reward(env, alarm: bool = True):
    combined_reward: CombinedScaledReward = env.get_reward_instance()
    combined_reward.addReward("Flat", IncreasingFlatReward(), 0.7)
    combined_reward.addReward("Redispatching", RedispReward(), 0.7)
    if alarm:
        combined_reward.addReward("Alarm", AlarmReward(), 0.3)
    else:
        print("\nWARNING: Alarm Reward deactivated\n")
    combined_reward.initialize(env)


def set_reward(env, config: RunConfiguration):
    combined_reward: CombinedScaledReward = env.get_reward_instance()
    grid2op_reward_module = importlib.import_module("grid2op.Reward")
    for (
        reward_name,
        reward_weight,
    ) in config.environment.reward.reward_components.items():
        combined_reward.addReward(
            reward_name,
            getattr(grid2op_reward_module, reward_name + "Reward")(),
            reward_weight,
        )

    combined_reward.initialize(env)


def evaluate(
    env: Union[
        grid2op.Environment.Environment, grid2op.Environment.MultiMixEnvironment
    ],
    agent: BaseAgent,
    path_save: str,
    nb_episode: int = 1,
    nb_process: int = 1,
    sequential: bool = False,
):
    Path(path_save).mkdir(parents=True, exist_ok=False)
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


def main(**kwargs):
    config = RunConfiguration(kwargs["run_file"])

    print("Running with env: " + config.environment.name)

    # Train Environment
    env_train = grid2op.make(
        config.environment.name + "_train80",
        reward_class=CombinedScaledReward,
        chronics_class=MultifolderWithCache,
        difficulty="competition",
    )
    set_reward(env_train, config)

    # Validation Environment
    # WARNING: chronics_class bugs the runner, don't set it in env_val
    env_val = grid2op.make(
        config.environment.name + "_val10",
        reward_class=CombinedScaledReward,
        difficulty="competition",
    )
    set_l2rpn_reward(env_val, alarm=False)

    # Set seed for reproducibility
    print("Running with seed: " + str(config.reproducibility.seed))
    fix_seed(env_train, env_val, seed=config.reproducibility.seed)

    if config.loading.load and Path(config.loading.load_dir).exists():
        agent = DPOP.load(
            log_file=config.loading.load_dir,
            env=env_train if config.training.train else env_val,
            tensorboard_dir=config.training.tensorboard_dir
            if config.training.train
            else None,
            checkpoint_dir=config.model.checkpoint_dir,
            name=config.model.name,
            training=config.training.train,
        )
    else:
        agent = DPOP(
            env=env_train,
            name=config.model.name,
            architecture=config.model.architecture,
            training=config.training.train,
            seed=config.reproducibility.seed,
            checkpoint_dir=config.model.checkpoint_dir,
            tensorboard_dir=config.training.tensorboard_dir,
            device=config.reproducibility.device,
        )

    if config.training.train:
        train(env_train, iterations=config.training.steps, dpop=agent)
    else:
        evaluate(
            env=env_val,
            agent=agent,
            path_save=config.evaluation.evaluation_dir,
            nb_episode=config.evaluation.episodes,
            sequential=True,
        )


p = argparse.ArgumentParser()
p.add_argument("--run-file", type=str)


if __name__ == "__main__":
    args = p.parse_args()
    main(**vars(args))
