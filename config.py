import json 
import os 

with open("config.json") as config_file:
    config = json.load(config_file)

os.environ["SIMILARITY_MODE"] = config["similarity_mode"]
os.environ["GOOGLE_API_KEY"] = config["google_api_key"]
os.environ["SERPER_API_KEY"] = config["serper_api_key"]

# read base prompt 
with open("base_prompt.txt", "r") as prompt_file:
    PROMPT = "".join( prompt_file.readlines() )