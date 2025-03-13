# Federated Learning with Flower

## Overview
This project implements a Federated Learning (FL) system using the [Flower](https://flower.ai) framework. It compares centralized and federated learning approaches on the CIFAR-10 dataset using PyTorch. Additionally, it incorporates **Differential Privacy (DP)** techniques to enhance security.

## Features
- **Federated Learning (FL):** Distributed model training using multiple clients.
- **Centralized Training:** Standard model training on a single dataset.
- **Differential Privacy (DP):** Adds noise to model updates for enhanced security.
- **Custom Model:** Convolutional Neural Network (CNN) for image classification.
- **Federated Simulation:** Uses Flower's simulation framework for testing FL models.
- **Performance Comparison:** Measures accuracy and training time for both centralized and federated models.
- **Configurable Hyperparameters:** Supports easy modification of training parameters via `config.json`.

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
├── config.py            # Stores device settings (CPU/GPU selection)
├── config.json          # Stores training hyperparameters
├── config_manager.py    # Handles config file updates
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
To run the project, navigate to the project directory and execute the following command:
```bash
cd code
python -m main
```
This command runs the `main.py` script, which iterates over different hyperparameter configurations and executes **federated learning experiments**.


## Configuration
Modify `config.json` to adjust hyperparameters:
```json
{
    "CENTRALIZED_EPOCHS": 100,
    "NUM_CLIENTS": 2,
    "NUM_ROUNDS": 1,
    "BATCH_SIZE": 8,
    "NOISE_MULTIPLIER": 0.1,
    "EVAL_RATIO": 0.3,
    "FIT_RATIO": 1.0
}
```
- `CENTRALIZED_EPOCHS`: Number of epochs for centralized training
- `NUM_CLIENTS`: Number of clients in federated learning
- `NUM_ROUNDS`: Number of federated training rounds
- `BATCH_SIZE`: Training batch size
- `NOISE_MULTIPLIER`: Level of differential privacy noise
- `EVAL_RATIO`: Ratio of clients used for evaluation

## Model Architecture
The CNN model (`model.py`) consists of convolutional, pooling, and fully connected layers designed for CIFAR-10 classification.

## Logging & Evaluation
The project logs accuracy results in CSV files for further analysis:
- **Centralized Training:** `log/centralized.csv`
- **Federated Learning:** `log/federated_acc.csv`

## Acknowledgments
- [Flower](https://flower.ai) for federated learning framework
- [PyTorch](https://pytorch.org/) for deep learning support

