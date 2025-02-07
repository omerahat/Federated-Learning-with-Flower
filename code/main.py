from federated_main import run_federated_learning  
from centralized_main import run_centralized_training

run_centralized_training()
print("\n---------------------------------\n")

history = run_federated_learning()

if history is not None and "accuracy" in history.metrics:
     final_accuracy = history.metrics["accuracy"][-1]
     print("Final aggregated accuracy:", final_accuracy)