import flwr as fl
from train import train, evaluate

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(), parameters))
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.trainloader)
        return self.get_parameters(config), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        accuracy = evaluate(self.model, self.valloader)
        return float(accuracy), len(self.valloader), {"accuracy": float(accuracy)}
