import torch  # PyTorch framework
import numpy as np  # NumPy for handling array operations
from typing import List  # Type hints for better code readability
from collections import OrderedDict  # Maintains the order of model parameters


def set_parameters(net, parameters: List[np.ndarray]):
    """Load model parameters into the network."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def get_parameters(net) -> List[np.ndarray]:
    """Extract model parameters as NumPy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


