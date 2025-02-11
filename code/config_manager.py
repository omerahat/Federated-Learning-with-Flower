import json

CONFIG_PATH = "config.json" 

def load_config(path=CONFIG_PATH):
     """Load configuration from a JSON file."""
     with open(path, "r") as f:
          data = json.load(f)
     return data

def update_config(key, value, path=CONFIG_PATH):
     """Update a specific configuration key with a new value."""
     config = load_config(path)
     config[key] = value
     with open(path, "w") as f:
          json.dump(config, f, indent=4)

