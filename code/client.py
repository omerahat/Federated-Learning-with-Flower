import torch  # PyTorch framework
import numpy as np  # NumPy for numerical operations
from flwr.client import NumPyClient  # Flower framework for federated learning
from typing import List  # Type annotations

# Import parameter handling functions
from utils import set_parameters, get_parameters  # Ensure these are defined


from model import Net  # Import the neural network model
from train import train, test  # Import training and evaluation functions

class FlowerClient(NumPyClient):
     def __init__(self, net, trainloader, valloader):
          self.net = net
          self.trainloader = trainloader
          self.valloader = valloader

     def get_parameters(self, config):
          return get_parameters(self.net)

     def fit(self, parameters, config):
          set_parameters(self.net, parameters)
          train(self.net, self.trainloader, epochs=1)
          return get_parameters(self.net), len(self.trainloader), {}

     def evaluate(self, parameters, config):
          set_parameters(self.net, parameters)
          loss, accuracy = test(self.net, self.valloader)
          return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

