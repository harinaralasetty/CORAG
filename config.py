import json 
import os 

APP_TITLE = "Retrieval Augmented Generation on Any Machine"
DESCRIPTION = "Upload a PDF document and ask your question. Answers will be retrieved from the document and enhanced by the Gemini large language model."

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"] = ""
SERPER_API_KEY = os.environ["SERPER_API_KEY"] = ""

HNSW_CONFIG = {
    "ef_construction": 500,
    "M": 30
}

# read base prompt 
with open("base_prompt.txt", "r") as prompt_file:
    PROMPT = "".join( prompt_file.readlines() )