import os 

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"] = ""
SERPER_API_KEY = os.environ["SERPER_API_KEY"] = ""

HNSW_CONFIG = {
    "ef_construction": 500,
    "M": 30
}

CHUNK_SETTINGS = {
    "CHUNK_SIZE": 200, 
    "CHUNK_OVERLAP": 50
}

# read base prompt 
with open("prompts/base_prompt.txt", "r") as prompt_file:
    BASE_PROMPT = "".join( prompt_file.readlines() )

# read query rephrasal prompt 
with open("prompts/query_rephrasal_prompt.txt", "r") as prompt_file:
    QUERY_REPHRASAL_PROMPT = "".join( prompt_file.readlines() )

# read chat_naming prompt 
with open("prompts/chat_naming.txt", "r") as prompt_file:
    CHAT_NAMING_PROMPT = "".join( prompt_file.readlines() )

# chat settings
LAST_N = 2 # last n messages to send to the llm 

# theme length
THEME_LENGTH = 2000 # first n characters to determine theme

models_list = ["gemini-pro", "gemini-1.5-flash", "gemini-exp-1206", "gemini-2.0-flash-exp", "gemini-2.0-flash-thinking-exp-1219"]