from typing import Any, Dict, Optional
from grid2op.Agent.agentWithConverter import AgentWithConverter
from grid2op.Environment.BaseEnv import BaseEnv
from grid2op.Agent.recoPowerlineAgent import RecoPowerlineAgent
from grid2op.Observation.baseObservation import BaseObservation
from pop.configs.architecture import Architecture
from pop.multiagent_system.dpop import DPOP
from pop.multiagent_system.space_factorization import EncodedAction
from pop.networks.serializable_module import SerializableModule


class ExpertPop(SerializableModule, AgentWithConverter):
    def __init__(
        self, pop: DPOP, checkpoint_dir: str, expert_only: bool = False
    ) -> None:
        super().__init__(log_dir=checkpoint_dir, name=pop.name)
        self.greedy_reconnect_agent = RecoPowerlineAgent(pop.env.action_space)
        self.safe_max_rho = pop.architecture.pop.safe_max_rho
        self.curtail_storage_limit = pop.architecture.pop.curtail_storage_limit
        self.expert_steps = 0
        self.step_pop = False
        self.pop = pop
        self.expert_only = expert_only

    def my_act(self, observation, reward, done=False):
        self.expert_steps += 1
        reconnection_action = self.greedy_reconnect_agent.act(observation, reward)

        if reconnection_action.impact_on_objects()["has_impact"]:
            # If there is some powerline to reconnect do it
            self.step_dpop = False
            action = reconnection_action

        elif not self.expert_only and max(observation.rho) > self.safe_max_rho:
            # If there is some powerline overloaded ask the agent what to do
            self.step_dpop = True
            action = self.pop.act(observation, reward, done)
            action.limit_curtail_storage(observation, margin=self.curtail_storage_limit)

        else:
            # else do nothing
            self.step_dpop = False
            action = self.pop.env.action_space({})

        self.pop.writer.add_text("Expert Action", str(action), self.expert_steps)
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
        self.pop.log_reward(reward, self.expert_steps, name="Expert Reward")

        if self.step_dpop:
            self.pop.step(action, observation, reward, next_observation, done)
        else:
            self.pop.alive_steps += 1
            if done:
                self.pop.log_alive_steps(self.pop.alive_steps, self.pop.episodes)
                self.pop.episodes += 1
                self.pop.alive_steps = 0

    def get_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = self.pop.get_state()
        state["expert_steps"] = self.expert_steps
        return state

    @property
    def episodes(self):
        return self.pop.episodes

    @property
    def writer(self):
        return self.pop.writer

    @property
    def train_steps(self):
        return self.expert_steps

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

        pop = DPOP.factory(
            checkpoint,
            env,
            tensorboard_dir,
            checkpoint_dir,
            name,
            training,
            local,
            pre_train,
            reset_exploration,
            architecture,
        )
        expert_pop = ExpertPop(
            pop,
            checkpoint_dir
            if checkpoint_dir is not None
            else checkpoint["checkpoint_dir"],
        )
        expert_pop.expert_steps = checkpoint["expert_steps"]
        return expert_pop
