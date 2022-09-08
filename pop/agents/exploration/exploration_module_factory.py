import importlib

from agents.exploration.exploration_module import ExplorationModule


def get_exploration_module(agent) -> ExplorationModule:
    from agents.base_gcn_agent import BaseGCNAgent

    agent: BaseGCNAgent
    exploration_module = agent.architecture.exploration.method
    exploration_module_class = "".join(
        [s.capitalize() for s in exploration_module.split("_")]
    )
    cls = getattr(
        importlib.import_module(__package__ + ".modules." + exploration_module),
        exploration_module_class,
    )
    return cls(agent)
