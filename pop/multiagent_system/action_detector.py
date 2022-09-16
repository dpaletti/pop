import networkx as nx


class ActionDetector:
    def __init__(self, n_actions: int, loop_length: int = 1):
        if loop_length < 1:
            return
        self.action_graph_memory: nx.DiGraph = nx.DiGraph()
        self.action_graph_memory.add_nodes_from(list(range(n_actions)))
        self.action_graph_memory.add_node(-1)

        self.last_action: int = -1

        self.loop_length: int = loop_length

    def is_repeated(self, action: int) -> bool:
        if self.loop_length < 1:
            return False

        copied_action_graph_memory: nx.DiGraph = self.action_graph_memory.copy()
        copied_action_graph_memory.add_edge(self.last_action, action)
        if any(
            filter(
                lambda cycle: len(cycle) <= self.loop_length,
                nx.simple_cycles(copied_action_graph_memory),
            )
        ):
            return True

        self.action_graph_memory = copied_action_graph_memory
        self.last_action = action
        return False
