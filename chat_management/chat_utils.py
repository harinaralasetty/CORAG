import os
import json
from typing import Dict, Any

from config import DATA_DIR

THREADS_INFO_FILE = os.path.join(DATA_DIR, "chat_threads_info.json")


def ensure_data_directory_exists() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def load_chat_threads_info() -> Dict[str, Any]:
    ensure_data_directory_exists()
    if not os.path.isfile(THREADS_INFO_FILE):
        return {}
    try:
        with open(THREADS_INFO_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_chat_threads_info(chat_threads_info: Dict[str, Any]) -> None:
    ensure_data_directory_exists()
    with open(THREADS_INFO_FILE, "w", encoding="utf-8") as f:
        json.dump(chat_threads_info, f, indent=4)


def format_exchange(message):
    return f"User: {message['user']} \nAssistant: {message['answer']}\n"


def get_chat_history(chat_history, skip_last=False):
    """Format a thread's exchanges; skip_last leaves out the exchange currently being answered."""
    messages = chat_history['messages'][:-1] if skip_last else chat_history['messages']
    return [format_exchange(m) for m in messages]

