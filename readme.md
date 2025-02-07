# Federated Learning with Flower

## Overview
This project implements a Federated Learning (FL) system using the [Flower](https://flower.ai) framework. It compares centralized and federated learning approaches on the CIFAR-10 dataset using PyTorch.

## Features
- **Federated Learning (FL):** Distributed model training using multiple clients.
- **Centralized Training:** Standard model training on a single dataset.
- **Custom Model:** Convolutional Neural Network (CNN) for image classification.
- **Federated Simulation:** Uses Flower's simulation framework for testing FL models.
- **Performance Comparison:** Measures accuracy and training time for both centralized and federated models.

## Project Structure
```
.
├── client.py            # Implements the Flower client
├── server.py            # Configures the federated learning server
├── federated_main.py    # Runs federated learning experiments
├── centralized_main.py  # Runs centralized training experiments
├── data_loader.py       # Loads and processes CIFAR-10 dataset
├── model.py             # Defines the CNN model
├── train.py             # Implements training and evaluation functions
├── config.py            # Stores project settings (e.g., number of clients, epochs)
├── utils.py             # Helper functions for model parameter handling
├── main.py              # Compares centralized and federated training
└── README.md            # Project documentation
```

## Installation
### 1. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Project
To run the project, navigate to the `code` directory using the appropriate command for your operating system:
- **Windows:** `cd code`
  
Then, execute the following command:
```bash
python -m main
```
This command runs the `main` module, which compares centralized and federated training approaches.

## Configuration
Modify `config.py` to adjust hyperparameters:
```python
CENTRALIZED_EPOCH = 30  # Number of epochs for centralized training
NUM_CLIENTS = 50        # Number of federated clients
NUM_ROUNDS = 20         # Rounds of federated learning
BATCH_SIZE = 128        # Batch size for training
```

## Model Architecture
The CNN model (`model.py`) consists of convolutional, pooling, and fully connected layers for classifying CIFAR-10 images.

## Acknowledgments
- [Flower](https://flower.ai) for federated learning framework
- [PyTorch](https://pytorch.org/) for deep learning support


