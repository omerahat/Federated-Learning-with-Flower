import flwr as fl
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.common import Context


def weighted_average(metrics):
    total_samples = sum([num_samples for _, num_samples in metrics])
    return sum([metric * num_samples for metric, num_samples in metrics]) / total_samples


def server_fn(context: Context) -> ServerAppComponents:
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=10,
        min_evaluate_clients=5,
        min_available_clients=10,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    config = ServerConfig(num_rounds=5)
    return ServerAppComponents(strategy=strategy, config=config)
