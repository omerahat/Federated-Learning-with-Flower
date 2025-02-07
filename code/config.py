import torch
CENTRALIZED_EPOCH = 25
NUM_CLIENTS = 50
NUM_ROUNDS = 4
BATCH_SIZE = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BACKEND_CONFIG = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}

if DEVICE.type == "cuda":
    BACKEND_CONFIG = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}
