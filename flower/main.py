from model import Net
from data_loader import load_data
from client import FlowerClient
import flwr as fl

trainloader, valloader = load_data()
model = Net()

fl.client.start_numpy_client(server_address="localhost:8080", client=FlowerClient(model, trainloader, valloader))
