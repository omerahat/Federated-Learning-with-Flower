import flwr as fl
from model import Net
from data_loader import load_data
from client import FlowerClient
from server import server_fn
import time

# Define a function to create clients
def client_fn(cid: str):
    model = Net()
    trainloader, valloader = load_data()
    return FlowerClient(model, trainloader, valloader)

# Simulation configuration
NUM_CLIENTS = 10
ROUNDS = 5

# Run the simulation
if __name__ == "__main__":
    start_time = time.time()

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=ROUNDS)
    )

    end_time = time.time()
    print(f"Federated Learning Simulation Time: {end_time - start_time:.2f} seconds")
