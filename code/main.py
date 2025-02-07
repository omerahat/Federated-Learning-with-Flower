from federated_main import run_federated_learning  
from centralized_main import run_centralized_training
import time

epochs = 20
t1 = time.time()
centralized_acc =  run_centralized_training(epochs=epochs)
t2 = time.time()

print("\n---------------------------------\n")

t3 = time.time()
history = run_federated_learning()
t4 = time.time()

if history is not None and "accuracy" in history.metrics:
     final_accuracy = history.metrics["accuracy"][-1]
     print("Final aggregated accuracy:", final_accuracy)

print(f"\nCentralized training took {t2-t1:.4f} seconds")
print(centralized_acc)

print(f"Federated learning took {t4-t3:.4f} seconds")
print(final_accuracy)