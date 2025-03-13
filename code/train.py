# Core PyTorch imports
import torch
import torch.nn as nn  # For CrossEntropyLoss
import torch.optim as optim  # For Adam optimizer

# Neural network components
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

# Tensor operations
from torch import max as torch_max  # Used in accuracy calculation
from torch import no_grad  # For evaluation context

# Device management (CPU/GPU)
from config import DEVICE  # For device management

def train(net, trainloader, epochs: int, verbose=False):
     """Train the network on the training set."""
     criterion = torch.nn.CrossEntropyLoss()
     optimizer = torch.optim.Adam(net.parameters())
     net.train()
     for epoch in range(epochs):
          correct, total, epoch_loss = 0, 0, 0.0
          for batch in trainloader:
               images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
               optimizer.zero_grad()
               outputs = net(images)
               loss = criterion(outputs, labels)
               loss.backward()
               optimizer.step()
               # Metrics
               epoch_loss += loss
               total += labels.size(0)
               correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
          epoch_loss /= len(trainloader.dataset)
          epoch_acc = correct / total
          if verbose:
               print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(net, testloader):
     """Evaluate the network on the entire test set."""
     criterion = torch.nn.CrossEntropyLoss()
     correct, total, loss = 0, 0, 0.0
     net.eval()
     with torch.no_grad():
          for batch in testloader:
               images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
               outputs = net(images)
               loss += criterion(outputs, labels).item()
               _, predicted = torch.max(outputs.data, 1)
               total += labels.size(0)
               correct += (predicted == labels).sum().item()
     loss /= len(testloader.dataset)
     accuracy = correct / total
     return loss, accuracy
