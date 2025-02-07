from data_loader import load_datasets
from train import train, test
from model import Net

from config import DEVICE


def run_centralized_training(epochs=5):
    """Run centralized training and evaluation on one data partition."""
    trainloader, valloader, testloader = load_datasets(partition_id=0)
    net = Net().to(DEVICE)

    # Train for 5 epochs
    for epoch in range(epochs):
        train(net, trainloader, 1)
        loss, accuracy = test(net, valloader)
        print(f"Epoch {epoch+1}: validation loss {loss}, accuracy {accuracy}")

    # Final test on the separate test set
    loss, accuracy = test(net, testloader)
    print(f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}")
    return accuracy


