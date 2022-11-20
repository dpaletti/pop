import argparse
import importlib
import logging
import os
import random
import warnings
from pathlib import Path
from typing import Optional, Union

import dgl
import grid2op.Environment
from grid2op.Reward.BaseReward import BaseReward
from grid2op.Reward.FlatReward import FlatReward
from grid2op.utils.l2rpn_2020_scores import ScoreL2RPN2020
import numpy as np
import pandas as pd
import torch as th
from configs.run_config import RunConfiguration
from grid2op import Environment
from grid2op.Action import BaseAction
from grid2op.Agent import BaseAgent, DoNothingAgent
from grid2op.Chronics import MultifolderWithCache
from grid2op.Environment import BaseEnv
from grid2op.Reward import (
    AlarmReward,
    CombinedReward,
    CombinedScaledReward,
    IncreasingFlatReward,
    RedispReward,
    EpisodeDurationReward,
)
from grid2op.Runner import Runner
from grid2op.utils import ScoreL2RPN2022, ScoreL2RPN2020

from pop.multiagent_system.base_pop import train
from pop.multiagent_system.dpop import DPOP
from pop.constants import PER_PROCESS_GPU_MEMORY_FRACTION
import re

from pop.multiagent_system.expert_pop import ExpertPop
from lightsim2grid import LightSimBackend

logging.getLogger("lightning").addHandler(logging.NullHandler())
logging.getLogger("lightning").propagate = False

warnings.filterwarnings("ignore", category=UserWarning)


class DQNReward(BaseReward):
    def __init__(self, per_step: int = 1):
        super(BaseReward, self).__init__()
        self._per_step = per_step
        self.reward_min = -1
        self.reward_max = self._per_step

    def __call__(
        self,
        action: BaseAction,
        env: Environment,
        has_error: bool,
        is_done: bool,
        is_illegal: bool,
        is_ambiguous: bool,
    ):
        super().__call__(
            action=action,
            env=env,
            has_error=has_error,
            is_done=is_done,
            is_illegal=is_illegal,
            is_ambiguous=is_ambiguous,
        )
        return self.reward_min if has_error else self.reward_max


# def set_experimental_reward(env):
# combined_reward: CombinedReward = env.get_reward_instance()
# combined_reward.addReward(
# "redisp",
# RedispReward.generate_class_custom_params(alpha_redisph=10, min_reward=-1)(),
# )
# combined_reward.addReward("flat", IncreasingFlatReward(per_timestep=1))
# combined_reward.initialize(env)


def set_experimental_reward(env):
    combined_reward: CombinedReward = env.get_reward_instance()
    combined_reward.addReward("flat", FlatReward(per_timestep=5))
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
    sequential: bool = True,
    do_nothing=False,
    score2020=False,
):
    if do_nothing:
        agent = DoNothingAgent(env.action_space)
    if not Path(path_save).exists():
        Path(path_save).mkdir(parents=True, exist_ok=False)

    os.environ[Runner.FORCE_SEQUENTIAL] = "1"
    nb_process = 1

    if config.evaluation.compute_score:
        csv_path = Path(
            path_save,
            ("l2rpn_2022_score_" if not score2020 else "l2rpn_2020_score_")
            + str(config.environment.difficulty)
            + ".csv",
        )
        if csv_path.exists():
            return pd.read_csv(csv_path)
        print("Created Directory: " + str(csv_path))
        score_func = ScoreL2RPN2022 if not score2020 else ScoreL2RPN2020
        score = score_func(
            env, nb_scenario=nb_episode, verbose=2, nb_process_stats=nb_process
        )
        agent_score = score.get(agent)
        agent_score_df = pd.DataFrame(agent_score).transpose()
        agent_score_df.columns = ["all_scores", "ts_survived", "total_ts"]
        agent_score_df.to_csv(csv_path)

    if config.evaluation.generate_grid2viz_data:
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
    # th.use_deterministic_algorithms(True)
    th.cuda.manual_seed_all(seed)
    dgl.seed(seed)


def main(**kwargs):

    config = RunConfiguration(kwargs["run_file"])

    if th.cuda.is_available():
        th.cuda.memory.set_per_process_memory_fraction(PER_PROCESS_GPU_MEMORY_FRACTION)

    print(
        "Running with env: "
        + config.environment.name
        + " with difficulty "
        + str(config.environment.difficulty)
    )
    reward = {
        "Episode Duration": EpisodeDurationReward(per_timestep=1 / 20),
        "Flat": FlatReward(per_timestep=5),
        "DQNReward": DQNReward(per_step=1),
    }

    # reward_class = (
    # CombinedReward
    # if list(config.environment.reward.reward_components.items())[0][0]
    # == "Experimental"
    # else CombinedScaledReward
    # )

    # Train Environment
    env_train = grid2op.make(
        config.environment.name + "_train80",
        chronics_class=MultifolderWithCache,
        difficulty=config.environment.difficulty,
        reward_class=reward[config.environment.reward],
        backend=LightSimBackend(),
    )

    # if reward_class == CombinedScaledReward:
    #    set_reward(env_train, config)
    # else:
    #    set_experimental_reward(env_train)

    # Validation Environment
    # WARNING: chronics_class bugs the runner, don't set it in env_val
    env_val = grid2op.make(
        config.environment.name + "_val10",
        difficulty=config.environment.difficulty,
        backend=LightSimBackend(),
    )
    # set_l2rpn_reward(env_val, alarm=False)

    # Set seed for reproducibility
    print("Running with seed: " + str(config.reproducibility.seed))
    fix_seed(env_train, env_val, seed=config.reproducibility.seed)

    if config.model.architecture.pop.enable_expert:
        agentType = ExpertPop
    else:
        agentType = DPOP

    if (
        config.loading.load
        and Path(config.loading.load_dir).parents[0].exists()
        and len(list(Path(config.loading.load_dir).parents[0].iterdir())) > 0
        and not config.model.do_nothing
    ):

        print("Loading " + config.model.name + " from " + config.loading.load_dir)
        agent = agentType.load(
            log_file=config.loading.load_dir,
            env=env_train if config.training.train else env_val,
            tensorboard_dir=config.training.tensorboard_dir,
            checkpoint_dir=config.model.checkpoint_dir,
            name=config.model.name,
            training=config.training.train,
            local=config.training.local,
            pre_train=config.training.pre_train,
            reset_exploration=config.loading.reset_exploration,
            architecture=config.model.architecture,
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
            local=config.training.local,
            pre_train=config.training.pre_train,
            feature_ranges=config.environment.feature_ranges,
        )

        if config.model.architecture.pop.enable_expert:
            agent = ExpertPop(
                pop=agent,
                checkpoint_dir=config.model.checkpoint_dir,
                expert_only=config.model.expert_only,
            )

    if config.training.train:

        if config.training.chronics == -1:
            print("Loading chronics up to 1000...")
            env_train.chronics_handler.set_filter(
                lambda path: re.match(".*[0-9][0-9][0-9].*", path) is not None
            )
        elif config.training.chronics == 10:
            print("Loading chronics up to 10...")
            env_train.chronics_handler.set_filter(
                lambda path: re.match(".*00[0-9].*", path) is not None
            )
        elif config.training.chronics == 100:
            print("Loading chronics up to 100...")
            env_train.chronics_handler.set_filter(
                lambda path: re.match(".*0[0-9][0-9].*", path) is not None
            )
        else:
            print("Loading chronics matching " + str(config.training.chronics))
            env_train.chronics_handler.set_filter(
                lambda path: re.match(".*" + str(config.training.chronics) + ".*", path)
                is not None
            )

        kept = env_train.chronics_handler.real_data.reset()
        print(
            "Loaded " + str(len(kept)) + " chronics. Training " + str(config.model.name)
        )
        # yappi.set_clock_type("cpu")
        # yappi.start(builtins=True)
        train(
            env_train,
            iterations=config.training.steps - agent.train_steps,
            dpop=agent,
            save_frequency=config.training.save_frequency,
        )
        # stats = yappi.get_func_stats()
        # stats.save("yappi_out", type="callgrind")
    else:
        print("Evaluating " + str(config.model.name))
        evaluate(
            config=config,
            env=env_val,
            agent=agent,
            path_save=config.evaluation.evaluation_dir,
            nb_episode=config.evaluation.episodes,
            sequential=True,
            do_nothing=config.model.do_nothing,
            score2020=True if config.evaluation.score == "2020" else False,
        )


p = argparse.ArgumentParser()
p.add_argument("--run-file", type=str)


if __name__ == "__main__":
    args = p.parse_args()
    main(**vars(args))
