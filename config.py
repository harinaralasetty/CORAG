import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
SEARCH_PROVIDER = os.environ.get("SEARCH_PROVIDER", "serpapi")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

EMBEDDING_PROVIDERS = ["google", "voyage", "openai", "local"]

HNSW_CONFIG = {
    "ef_construction": 500,
    "M": 30
}

# characters per chunk
CHUNK_SETTINGS = {
    "CHUNK_SIZE": 1000,
    "CHUNK_OVERLAP": 200
}

# where chats and their vector indexes are stored (survives restarts)
DATA_DIR = os.environ.get("CORAG_DATA_DIR", "data")

PROMPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")


def _read_prompt(name):
    with open(os.path.join(PROMPT_DIR, name), "r", encoding="utf-8") as prompt_file:
        return prompt_file.read()


BASE_PROMPT = _read_prompt("base_prompt.txt")
QUERY_REPHRASAL_PROMPT = _read_prompt("query_rephrasal_prompt.txt")
CHAT_NAMING_PROMPT = _read_prompt("chat_naming.txt")

# chat settings
LAST_N = 2 # last n messages to send to the llm 

# theme length
THEME_LENGTH = 2000 # first n characters to determine theme

models_list = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-flash-latest",
    "gemini-pro-latest",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
]


def default_model():
    """First Gemini model when a Google key exists, otherwise a Claude model if that key exists."""
    if not GOOGLE_API_KEY and ANTHROPIC_API_KEY:
        return next(m for m in models_list if m.startswith("claude"))
    return models_list[0]


def default_embedding_provider():
    """Pick the first provider whose key is set; local needs no key."""
    if GOOGLE_API_KEY:
        return "google"
    if VOYAGE_API_KEY:
        return "voyage"
    if OPENAI_API_KEY:
        return "openai"
    return "local"


def has_any_llm_key():
    return bool(GOOGLE_API_KEY or ANTHROPIC_API_KEY)
