import grid2op
from pop.community_detector import CommunityDetector
import networkx as nx

def test_community_detector():
    nm_env = "l2rpn_icaps_2021_small"
    env = grid2op.make(nm_env)
    obs1 = env.reset()
    graph1 = obs1.as_networkx()
    community_detector = CommunityDetector(0)
    community1 = community_detector.dynamo(graph1)
    print("\n\nCommunity intialization:")
    print(community1)
    print("\n\n\n")
    obs2 = env.reset()
    graph2 = nx.Graph(obs2.as_networkx())
    print("Before Removal")
    print(graph2.edges)
    graph2.remove_edge(0, 4)
    print("After removal")
    print(graph2.edges)

    community2 = community_detector.dynamo(graph1, graph2, community1)
    print("Final Community")
    print(community2)
    graph3 = nx.Graph(graph2)
    graph3.add_edge(11, 35)
    community3 = community_detector.dynamo(graph2, graph3, community2)
    print(community3)
