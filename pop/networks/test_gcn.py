from configs.run_config import RunConfiguration
from networks.dueling_net import DuelingNet
import grid2op
from pop.node_agents.utilities import batch_observations


def test_gcn():
    run_configuration = RunConfiguration("../../run_files/dpop_rte_1e4.toml")
    embedding_architecture = run_configuration.model.architecture.agent.embedding
    advantage_architecture = run_configuration.model.architecture.agent.advantage_stream
    value_architecture = run_configuration.model.architecture.agent.value_stream
    g = DuelingNet(
        action_space_size=8,
        node_features=6,
        edge_features=14,
        name="some_name",
        embedding_architecture=embedding_architecture,
        advantage_stream_architecture=advantage_architecture,
        value_stream_architecture=value_architecture,
    )
    env = grid2op.make()
    obs1 = env.reset()
    obs2 = env.reset()

    batched = batch_observations((obs1, obs2), device="cpu")

    print(g(batched).shape)
