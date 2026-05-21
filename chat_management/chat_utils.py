import os
import json
import shutil
from typing import Dict, Any

TEMP_DIR = "temp_data"
THREADS_INFO_FILE = os.path.join(TEMP_DIR, "chat_threads_info.json")


def ensure_temp_directory_exists() -> None:
    os.makedirs(TEMP_DIR, exist_ok=True)


def load_chat_threads_info() -> Dict[str, Any]:
    ensure_temp_directory_exists()
    if not os.path.isfile(THREADS_INFO_FILE):
        return {}
    try:
        with open(THREADS_INFO_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_chat_threads_info(chat_threads_info: Dict[str, Any]) -> None:
    ensure_temp_directory_exists()
    with open(THREADS_INFO_FILE, "w", encoding="utf-8") as f:
        json.dump(chat_threads_info, f, indent=4)


def format_exchange(message):
    return f"User: {message['user']} \nAssistant: {message['answer']}\n"


def get_chat_history(chat_history):
    return [format_exchange(m) for m in chat_history['messages']]


def cleanup_temp_dir():
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
        print(f"Deleted temp directory: {TEMP_DIR}")
