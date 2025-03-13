from collections import OrderedDict
from typing import List, Tuple

from flwr.server import ServerApp
from flwr.simulation import run_simulation
from config import DEVICE, BACKEND_CONFIG
from client import FlowerClient
from server import server_fn
from flwr.client import Client, ClientApp
from model import Net
from flwr.common import Metrics, Context
from data_loader import load_datasets
from flwr.client.mod import adaptiveclipping_mod
from config_manager import load_config

def client_fn(context: Context) -> Client:
    """Create a Flower client representing a single organization."""

    # Load model
    net = Net().to(DEVICE)

    # Load data (CIFAR-10)
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data partition
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    trainloader, valloader, _ = load_datasets(partition_id=partition_id)

    # Create a single Flower client representing a single organization
    # FlowerClient is a subclass of NumPyClient, so we need to call .to_client()
    # to convert it to a subclass of `flwr.client.Client`
    return FlowerClient(net, trainloader, valloader).to_client()

def run_federated_learning():

    
    """Wrapper function to run the Flower simulation."""
    # Create the ClientApp and ServerApp
    client = ClientApp(client_fn=client_fn, mods=[adaptiveclipping_mod])
    server = ServerApp(server_fn=server_fn)

    # Run the Flower simulation
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=load_config()["NUM_CLIENTS"],
        backend_config=BACKEND_CONFIG,
    )


