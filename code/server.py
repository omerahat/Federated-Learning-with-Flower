import csv

from config_manager import load_config  # Load configuration
from typing import List, Tuple  # Type annotations
from flwr.server.strategy import FedAvg  # Federated Averaging strategy
from flwr.common import Metrics, Context  # Metric tracking and server context
from flwr.server import ServerConfig, ServerAppComponents  # Server setup
from flwr.server.strategy.dp_adaptive_clipping import DifferentialPrivacyClientSideAdaptiveClipping  


# CSV File Configuration
csv_file = r'C:\Users\ahato\Desktop\flowerProject\log\federated_acc.csv'
csv_cols = ["NUM_CLIENTS", "NUM_ROUNDS", "BATCH_SIZE", "NOISE_MULTIPLIER", "EVAL_RATIO", "ACCURACY"]

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    config_data= load_config()
    print("weighted_average function called!")
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    with open(csv_file, 'a', newline='') as f:  # ← **Fix: newline=''**
        writer = csv.writer(f)

        # Write the header only if the file is empty
        if f.tell() == 0:
            writer.writerow(csv_cols)    

        writer.writerow([config_data["NUM_CLIENTS"],
                        config_data["NUM_ROUNDS"],
                        config_data["BATCH_SIZE"],
                        config_data["NOISE_MULTIPLIER"],
                        config_data["EVAL_RATIO"], 
                        sum(accuracies) / sum(examples)]
                        )

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour."""
    config_data= load_config()
    # Create FedAvg strategy
    strategy = FedAvg(
        fraction_fit=config_data["FIT_RATIO"],
        fraction_evaluate=config_data["EVAL_RATIO"],
        min_fit_clients=int(config_data["FIT_RATIO"] * config_data["NUM_CLIENTS"]),
        min_evaluate_clients = max(1, int(config_data["EVAL_RATIO"] * config_data["NUM_CLIENTS"])),
        min_available_clients=int(config_data["NUM_CLIENTS"]),
        evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function     
    )

    dp_strategy = DifferentialPrivacyClientSideAdaptiveClipping(
        strategy=strategy,
        noise_multiplier=config_data["NOISE_MULTIPLIER"],
        num_sampled_clients=config_data["NUM_CLIENTS"],
        clipped_count_stddev=5,
        
    )

    # Configure the server for NUM_ROUNDS rounds
    config = ServerConfig(num_rounds=config_data["NUM_ROUNDS"])

    return ServerAppComponents(strategy=dp_strategy, config=config)

