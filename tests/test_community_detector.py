import grid2op
from pop.community_detector import CommunityDetector
import networkx as nx
import matplotlib.pyplot as plt


def test_community_detector():
    nm_env = "l2rpn_icaps_2021_small"
    env = grid2op.make(nm_env)
    obs1 = env.reset()
    graph1 = obs1.as_networkx()
    community_detector = CommunityDetector(0)
    community1_psm = community_detector.dynamo(
        graph1, enable_power_supply_modularity=True
    )
    community1 = community_detector.dynamo(graph1)
    print("\n\nCommunity intialization:")
    print("Power Supply Modularity")
    print(community1_psm)
    print("Modularity")
    print(community1)
    print("\n\n\n")

    obs2 = env.reset()
    graph2 = nx.Graph(obs2.as_networkx())
    graph2.remove_edge(0, 4)
    print("Removed (0, 4) edge")

    community2_psm = community_detector.dynamo(
        graph1, graph2, community1_psm, enable_power_supply_modularity=True
    )
    community2 = community_detector.dynamo(
        graph1, graph2, community1, enable_power_supply_modularity=True
    )
    print("Power Supply Modularity")
    print(community2_psm)
    print("Modularity")
    print(community2)
    graph3 = nx.Graph(graph2)
    graph3.add_edge(11, 35)
    print("Adding (11, 35) edge")
    community3_psm = community_detector.dynamo(graph2, graph3, community2_psm)
    community3 = community_detector.dynamo(graph2, graph3, community2)
    print("Power Supply  Modularity")
    print(community3_psm)
    print("Modularity")
    print(community3)


def community_to_dict(communities):
    out = {}

    for community, color in zip(communities, range(len(communities))):
        for node in community:
            out[node] = color / 10
    return out


def test_comm_viz():
    nm_env = "l2rpn_icaps_2021_small"
    env = grid2op.make(nm_env)
    obs1 = env.reset()
    graph1 = obs1.as_networkx()
    pos = nx.spring_layout(graph1)
    community_detector = CommunityDetector(0)
    community1 = community_detector.dynamo(graph1)
    community1_psm = community_detector.dynamo(
        graph1, enable_power_supply_modularity=True
    )
    print(community1)
    print(community1_psm)
    plt.figure("louvain", figsize=(8, 8))
    plt.axis("off")
    nx.draw_networkx_labels(graph1, pos)
    nx.draw_networkx_nodes(
        graph1,
        pos,
        node_size=600,
        cmap=plt.cm.RdYlBu,
        node_color=community_to_dict(community1),
    )
    nx.draw_networkx_edges(graph1, pos, alpha=0.3)
    plt.figure("psm", figsize=(8, 8))
    plt.axis("off")
    nx.draw_networkx_labels(graph1, pos)
    nx.draw_networkx_nodes(
        graph1,
        pos,
        node_size=600,
        cmap=plt.cm.RdYlBu,
        node_color=list(community_to_dict(community1_psm).values()),
    )

    nx.draw_networkx_edges(graph1, pos, alpha=0.3)
    plt.show()
    obs2 = env.reset()
    graph2 = nx.Graph(obs2.as_networkx())
    graph2.remove_edge(0, 4)
    print("Removed (0, 4) edge")

    community2_psm = community_detector.dynamo(
        graph1, graph2, community1_psm, enable_power_supply_modularity=True
    )
    community2 = community_detector.dynamo(
        graph1, graph2, community1, enable_power_supply_modularity=True
    )
    print(community2)
    print(community2_psm)
    print(list(community_to_dict(community2).values()))
    print(list(community_to_dict(community2_psm).values()))
    plt.figure("louvain removed (0, 4)", figsize=(8, 8))
    plt.axis("off")
    nx.draw_networkx_labels(graph2, pos)
    nx.draw_networkx_nodes(
        graph2,
        pos,
        label=list(graph2.nodes),
        node_size=600,
        cmap=plt.cm.RdYlBu,
        node_color=list(community_to_dict(community2).values()),
    )
    nx.draw_networkx_edges(graph2, pos, alpha=0.3)
    plt.figure("psm removed (0, 4)", figsize=(8, 8))
    plt.axis("off")
    nx.draw_networkx_labels(graph2, pos)
    nx.draw_networkx_nodes(
        graph2,
        pos,
        label=list(graph2.nodes),
        node_size=600,
        cmap=plt.cm.RdYlBu,
        node_color=list(community_to_dict(community2_psm).values()),
    )
    nx.draw_networkx_edges(graph2, pos, alpha=0.3)
    plt.show()

    graph3 = nx.Graph(graph2)
    graph3.add_edge(11, 35)
    graph3.add_edge(0, 4)
    print("Adding (11, 35), (0, 4) edge")
    community3_psm = community_detector.dynamo(graph2, graph3, community2_psm)
    community3 = community_detector.dynamo(graph2, graph3, community2)

    print(community3)
    print(community3_psm)
    plt.figure("louvain added (11, 35), (0, 4)", figsize=(8, 8))
    plt.axis("off")
    nx.draw_networkx_labels(graph3, pos)
    nx.draw_networkx_nodes(
        graph3,
        pos,
        label=list(graph3.nodes),
        node_size=600,
        cmap=plt.cm.RdYlBu,
        node_color=list(community_to_dict(community3).values()),
    )
    nx.draw_networkx_edges(graph3, pos, alpha=0.3)
    plt.figure("psm added (11 , 35), (0, 4)", figsize=(8, 8))
    plt.axis("off")
    nx.draw_networkx_labels(graph3, pos)
    nx.draw_networkx_nodes(
        graph3,
        pos,
        label=list(graph3.nodes),
        node_size=600,
        cmap=plt.cm.RdYlBu,
        node_color=list(community_to_dict(community3_psm).values()),
    )
    nx.draw_networkx_edges(graph3, pos, alpha=0.3)
    plt.show()
