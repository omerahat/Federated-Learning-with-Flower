from federated_main import run_federated_learning  
from centralized_main import run_centralized_training
from config_manager import update_config, load_config
import time
import csv

"""
NUM_CLIENTS_VALUES = [2]
NUM_ROUNDS_VALUES = [2]
BATCH_SIZE_VALUES = [128]
NOISE_MULTIPLIER_VALUES = [1.1]
EVAL_RATIO_VALUES = [0.3]
"""
update_config("CENTRALIZED_EPOCHS", 100)

run_centralized_training() !!!!!!!!!

# Load initial configuration
config = load_config()

# Define different parameter values to iterate over
NUM_CLIENTS_VALUES = [2, 5, 20, 50, 100, 200]
NUM_ROUNDS_VALUES = [1, 5, 10, 20,  50, 100]
BATCH_SIZE_VALUES = [32, 64, 128, 256]
NOISE_MULTIPLIER_VALUES = [0.1, 0.4, 1.1, 1.5, 2.5, 4]
EVAL_RATIO_VALUES = [0.1, 0.3, 0.75, 0.9, 1]


csv_file_exec = r'C:\Users\ahato\Desktop\flowerProject\log\execution_time.csv'
csv_cols_exec = ["NUM_CLIENTS", "NUM_ROUNDS", "BATCH_SIZE", "NOISE_MULTIPLIER", "EVAL_RATIO", "EXECUTION_TIME"]


with open(csv_file_exec, 'a', newline='') as f:  # ← **Fix: newline=''**
    writer = csv.writer(f)
    writer.writerow(csv_cols_exec)


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
                    time_start = time.time()
                    run_federated_learning()
                    time_end = time.time()
                    with open(csv_file_exec, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([num_clients, num_rounds, batch_size, noise_multiplier, eval_ratio, time_end - time_start])
