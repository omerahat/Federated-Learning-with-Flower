from typing import List, Tuple  # Type annotations
from flwr.server.strategy import FedAvg  # Federated Averaging strategy
from flwr.common import Metrics, Context  # Metric tracking and server context
from flwr.server import ServerConfig, ServerAppComponents  # Server setup
from config import NUM_ROUNDS
from flwr.server.strategy.dp_adaptive_clipping import DifferentialPrivacyClientSideAdaptiveClipping  

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
     # Multiply accuracy of each client by number of examples used
     accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
     examples = [num_examples for num_examples, _ in metrics]

     # Aggregate and return custom metric (weighted average)
     return {"accuracy": sum(accuracies) / sum(examples)}

def server_fn(context: Context) -> ServerAppComponents:
     """Construct components that set the ServerApp behaviour.

     You can use settings in `context.run_config` to parameterize the
     construction of all elements (e.g the strategy or the number of rounds)
     wrapped in the returned ServerAppComponents object.
     """

     # Create FedAvg strategy
     strategy = FedAvg(
          fraction_fit=1.0,
          fraction_evaluate=0.6,
          min_fit_clients=50,
          min_evaluate_clients=30,
          min_available_clients=50,
          evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function     
     )

     dp_strategy = DifferentialPrivacyClientSideAdaptiveClipping(
          strategy=strategy,
          noise_multiplier=1.1,
          num_sampled_clients=50,

     )

     # Configure the server for 5 rounds of training
     config = ServerConfig(num_rounds=NUM_ROUNDS)

     return ServerAppComponents(strategy=dp_strategy, config=config)