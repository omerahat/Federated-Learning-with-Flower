from data_loader import load_datasets
from train import train, test
from model import Net
from config import DEVICE
from config_manager import load_config

csv_file = r'C:\Users\ahato\Desktop\flowerProject\log\centralized.csv'



def run_centralized_training(epochs=load_config()["CENTRALIZED_EPOCHS"]):
    """Run centralized training and evaluation on one data partition."""
    trainloader, valloader, testloader = load_datasets(partition_id=0)
    net = Net().to(DEVICE)
    
          
    # Train for 5 epochs
    for epoch in range(epochs):
        train(net, trainloader, 1)
        loss, accuracy = test(net, valloader)
        with open(csv_file, 'a') as f:
            f.write(str(accuracy) + '\n')
        print(f"Epoch {epoch+1}: validation loss {loss}, accuracy {accuracy}")

    # Final test on the separate test set
    loss, accuracy = test(net, testloader)
    print(f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}")
    with open(csv_file, 'a') as f:
        f.write(str(accuracy) + '\n')
    return accuracy

