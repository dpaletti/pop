from typing import Any, Dict, Optional
from grid2op.Environment.BaseEnv import BaseEnv
from grid2op.Agent.recoPowerlineAgent import RecoPowerlineAgent
from grid2op.Observation.baseObservation import BaseObservation
from pop.agents.manager import Manager
from pop.agents.ray_gcn_agent import RayGCNAgent
from pop.agents.ray_shallow_gcn_agent import RayShallowGCNAgent
from pop.configs.architecture import Architecture
from pop.multiagent_system.dpop import DPOP
from pop.multiagent_system.space_factorization import EncodedAction

from pop.multiagent_system.dictatorship_penalizer import DictatorshipPenalizer
from tqdm import tqdm


class ExpertPop(DPOP):
    def __init__(
        self,
        env: BaseEnv,
        name: str,
        architecture: Architecture,
        training: bool,
        seed: int,
        checkpoint_dir: Optional[str] = ...,
        tensorboard_dir: Optional[str] = ...,
        device: Optional[str] = ...,
        local: bool = False,
        pre_train: bool = False,
    ) -> None:
        super().__init__(
            env,
            name,
            architecture,
            training,
            seed,
            checkpoint_dir,
            tensorboard_dir,
            device,
            local,
            pre_train,
        )
        self.greedy_reconnect_agent = RecoPowerlineAgent(env.action_space)
        self.safe_max_rho = architecture.pop.safe_max_rho
        self.expert_steps = -1
        self.step_dpop = False

    def my_act(self, observation, reward, done=False):
        self.expert_steps += 1
        reconnection_action = self.greedy_reconnect_agent.act(observation, reward)

        if reconnection_action.impact_on_objects()["has_impact"]:
            self.step_dpop = False
            # If there is some powerline to reconnect do it
            action = reconnection_action

        elif max(observation.rho) > self.safe_max_rho:
            print("Querying RL agent")
            self.step_dpop = True
            # If there is some powerline overloaded ask the agent what to do
            action = super().act(observation, reward, done)

        else:
            self.step_dpop = False
            # else do nothing
            action = self.env.action_space({})

        self.writer.add_text("Expert Action", str(action), self.expert_steps)
        return action

    def convert_act(self, action):
        return action

    def convert_obs(self, observation: BaseObservation) -> BaseObservation:
        return observation

    def step(
        self,
        action: EncodedAction,
        observation: BaseObservation,
        reward: float,
        next_observation: BaseObservation,
        done: bool,
    ):
        self.log_reward(reward, self.expert_steps, name="Expert Reward")

        if self.step_dpop:
            super().step(action, observation, reward, next_observation, done)
        else:
            self.alive_steps += 1
            if done:
                self.log_alive_steps(self.alive_steps, self.episodes)
                self.episodes += 1
                self.alive_steps = 0

    def get_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = super().get_state()
        state["expert_steps"] = self.expert_steps
        return state

    @staticmethod
    def factory(
        checkpoint: Dict[str, Any],
        env: Optional[BaseEnv] = None,
        tensorboard_dir: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        name: Optional[str] = None,
        training: Optional[bool] = None,
        local: bool = False,
        pre_train: bool = False,
        reset_exploration: bool = False,
        architecture: Optional[Architecture] = None,
    ) -> "ExpertPop":

        expert_pop: "ExpertPop" = ExpertPop(
            env=env,
            name=checkpoint["name"] if name is None else name,
            architecture=Architecture(load_from_dict=checkpoint["architecture"])
            if architecture is None
            else architecture,
            training=training,
            tensorboard_dir=tensorboard_dir,
            checkpoint_dir=checkpoint_dir,
            seed=checkpoint["seed"],
            device=checkpoint["device"],
            local=local,
            pre_train=pre_train,
        )

        # Differences with DPOP method
        expert_pop.expert_steps = checkpoint["expert_steps"]

        # Rest
        expert_pop.pre_initialized = True
        expert_pop.alive_steps = checkpoint["alive_steps"] if training else 0
        expert_pop.episodes = checkpoint["episodes"] if training else 0
        expert_pop.train_steps = checkpoint["train_steps"] if training else 0
        expert_pop.edge_features = checkpoint["edge_features"]
        expert_pop.node_features = checkpoint["node_features"]
        expert_pop.manager_initialization_threshold = checkpoint[
            "manager_initialization_threshold"
        ]
        expert_pop.community_update_steps = checkpoint["community_update_steps"]
        expert_pop.substation_to_agent = {
            sub_id: RayGCNAgent.load(
                checkpoint=agent_state,
                training=training,
                reset_exploration=reset_exploration,
                architecture=architecture.agent if architecture is not None else None,
            )
            if "optimizer_state" in list(agent_state.keys())
            else RayShallowGCNAgent.load(checkpoint=agent_state)
            for sub_id, agent_state in tqdm(checkpoint["agents_state"].items())
        }
        print("Loading Managers")
        expert_pop.managers_history = {
            Manager.load(
                checkpoint=manager_state,
                training=training,
                reset_exploration=reset_exploration,
                architecture=architecture.manager if architecture is not None else None,
            ): history
            for _, (manager_state, history) in tqdm(
                checkpoint["managers_state"].items()
            )
        }
        # TODO: managers here are treated as equal, they may be not

        if expert_pop.architecture.pop.incentives:
            for manager in expert_pop.managers_history.keys():
                expert_pop.manager_incentives.add_agent(manager, 1)

        if expert_pop.architecture.pop.dictatorship_penalty:
            for manager in expert_pop.managers_history.keys():
                expert_pop.manager_dictatorship_penalties[
                    manager
                ] = DictatorshipPenalizer(
                    choice_to_ranking={
                        substation: len(action_converter.all_actions)
                        for substation, action_converter in expert_pop.substation_to_action_converter.items()
                    },
                    **expert_pop.architecture.pop.dictatorship_penalty
                )

        print("Loading Head Manager")
        expert_pop.head_manager = Manager.load(
            checkpoint=checkpoint["head_manager_state"],
            training=training,
            reset_exploration=reset_exploration,
            architecture=architecture.head_manager
            if architecture is not None
            else None,
        )
        print("Loading is over")
        return expert_pop
