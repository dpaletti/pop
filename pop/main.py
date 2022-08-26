from pathlib import Path

import yappi
from grid2op import Environment
from grid2op.Action import BaseAction
from grid2op.Agent import BaseAgent
from grid2op.Environment import BaseEnv
from grid2op.Reward import (
    CombinedScaledReward,
    RedispReward,
    IncreasingFlatReward,
    AlarmReward,
    CombinedReward,
)
from grid2op.Chronics import MultifolderWithCache
from typing import Union, Optional
import grid2op.Environment
from grid2op.Runner import Runner
import torch as th
import random
import numpy as np
import dgl
import os

from grid2op.utils import ScoreL2RPN2022

from configs.run_config import RunConfiguration
import argparse
import importlib

import warnings

from pop.multiagent_system.dpop import DPOP
from pop.multiagent_system.base_pop import train
import pandas as pd
import cProfile

warnings.filterwarnings("ignore", category=UserWarning)


class NoActionRedispReward(RedispReward):
    def __init__(self, logger=None):
        super(NoActionRedispReward, self).__init__(logger=logger)
        self.previous_action: Optional[BaseAction] = None

    def __call__(
        self,
        action: BaseAction,
        env: Environment,
        has_error: bool,
        is_done: bool,
        is_illegal: bool,
        is_ambiguous: bool,
    ):
        reward = super().__call__(
            action=action,
            env=env,
            has_error=has_error,
            is_done=is_done,
            is_illegal=is_illegal,
            is_ambiguous=is_ambiguous,
        )
        return reward


def set_experimental_reward(env):
    combined_reward: CombinedReward = env.get_reward_instance()
    combined_reward.addReward(
        "redisp",
        RedispReward.generate_class_custom_params(alpha_redisph=10, min_reward=-1)(),
    )
    # combined_reward.addReward("episode", EpisodeDurationReward(per_timestep=1000))
    combined_reward.addReward("flat", IncreasingFlatReward(per_timestep=1))
    combined_reward.initialize(env)


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
    config: RunConfiguration,
    env: Union[
        grid2op.Environment.Environment, grid2op.Environment.MultiMixEnvironment
    ],
    agent: BaseAgent,
    path_save: str,
    nb_episode: int = 1,
    nb_process: int = 1,
    sequential: bool = False,
):
    if not Path(path_save).exists():
        Path(path_save).mkdir(parents=True, exist_ok=False)
    if config.evaluation.compute_score:
        csv_path = Path(
            path_save, "l2rpn_2022_score_" + str(config.environment.difficulty) + ".csv"
        )
        if csv_path.exists():
            return pd.read_csv(csv_path)
        score = ScoreL2RPN2022(env, nb_scenario=nb_episode, verbose=1)
        agent_score = score.get(agent)
        agent_score_df = pd.DataFrame(agent_score).transpose()
        agent_score_df.columns = ["all_scores", "ts_survived", "total_ts"]
        agent_score_df.to_csv(csv_path)

    if config.evaluation.generate_grid2viz_data:
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

    print(
        "Running with env: "
        + config.environment.name
        + " with difficulty "
        + str(config.environment.difficulty)
    )

    reward_class = (
        CombinedReward
        if list(config.environment.reward.reward_components.items())[0][0]
        == "Experimental"
        else CombinedScaledReward
    )

    # Train Environment
    env_train = grid2op.make(
        config.environment.name + "_train80",
        reward_class=reward_class,
        chronics_class=MultifolderWithCache,
        difficulty=config.environment.difficulty,
    )
    if reward_class == CombinedScaledReward:
        set_reward(env_train, config)
    else:
        set_experimental_reward(env_train)

    # Validation Environment
    # WARNING: chronics_class bugs the runner, don't set it in env_val
    env_val = grid2op.make(
        config.environment.name + "_val10",
        reward_class=CombinedScaledReward,
        difficulty=config.environment.difficulty,
    )
    set_l2rpn_reward(env_val, alarm=False)

    # Set seed for reproducibility
    print("Running with seed: " + str(config.reproducibility.seed))
    fix_seed(env_train, env_val, seed=config.reproducibility.seed)

    if config.loading.load and Path(config.loading.load_dir).parents[0].exists():
        print("Loading model from last checkpoint...")
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
        print("Training...")
        yappi.set_clock_type("cpu")
        yappi.start(builtins=True)
        train(env_train, iterations=config.training.steps, dpop=agent)
        stats = yappi.get_func_stats()
        stats.save("yappi_out", type="callgrind")
    else:
        print("Evaluating...")
        evaluate(
            config=config,
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
