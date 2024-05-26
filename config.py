import json 
import os 

with open("config.json") as config_file:
    config = json.load(config_file)

os.environ["SIMILARITY_MODE"] = config["similarity_mode"]
os.environ["GOOGLE_API_KEY"] = config["google_api_key"]