from federated_main import run_federated_learning  
from centralized_main import run_centralized_training
from config_manager import update_config, load_config
"""
NUM_CLIENTS_VALUES = [2, 3, 5, 10, 20, 50, 100, 200]
NUM_ROUNDS_VALUES = [2, 5, 8, 10, 15, 20, 30, 50]
BATCH_SIZE_VALUES = [4, 8, 16, 32, 64, 128, 256]
NOISE_MULTIPLIER_VALUES = [1.1, 1.25, 1.5, 1.8, 2.0, 2.5, 4]
EVAL_RATIO_VALUES = [0.1, 0.25, 0.4, 0.6, 0.75, 0.9, 1.0]
"""

update_config("CENTRALIZED_EPOCHS", 100)
run_centralized_training()

# Load initial configuration
config = load_config()

# Define different parameter values to iterate over
NUM_CLIENTS_VALUES = [2, 3, 5, 10, 20, 50, 100, 200]
NUM_ROUNDS_VALUES = [1, 2, 5, 8, 10, 20,  50]
BATCH_SIZE_VALUES = [8, 16, 32, 64, 128, 256]
NOISE_MULTIPLIER_VALUES = [0.1, 0.4, 1.1, 1.5, 2.0, 2.5, 4]
EVAL_RATIO_VALUES = [0.1, 0.3, 0.6, 0.75, 0.8, 0.9]

# Update configuration for each combination and run learning
for num_clients in NUM_CLIENTS_VALUES:
    update_config("NUM_CLIENTS", num_clients)
    for num_rounds in NUM_ROUNDS_VALUES:
        update_config("NUM_ROUNDS", num_rounds)
        for batch_size in BATCH_SIZE_VALUES:
            update_config("BATCH_SIZE", batch_size)
            for noise_multiplier in NOISE_MULTIPLIER_VALUES:
                update_config("NOISE_MULTIPLIER", noise_multiplier)
                for eval_ratio in EVAL_RATIO_VALUES:
                    update_config("EVAL_RATIO", eval_ratio)
                    print("NUM_CLIENTS:", num_clients, 
                          "NUM_ROUNDS:", num_rounds, 
                          "BATCH_SIZE:", batch_size, 
                          "NOISE_MULTIPLIER:", noise_multiplier, 
                          "EVAL_RATIO:", eval_ratio)
                    run_federated_learning()