import json

# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------
def load_config(): 
    # Open and load the JSON file
    with open('utils/config.json', 'r') as file:
        config = json.load(file)
    return config