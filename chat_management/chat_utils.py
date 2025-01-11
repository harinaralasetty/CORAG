
import os
import json
from typing import Dict, Any
import shutil
import os 

TEMP_DIR = "temp_data"
THREADS_INFO_FILE = os.path.join(TEMP_DIR, "chat_threads_info.json")

def ensure_temp_directory_exists() -> None:
    """
    Ensures that the temp directory exists. Creates it if not found.
    """
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

def load_chat_threads_info() -> Dict[str, Any]:
    """
    Loads the JSON file containing basic info for all threads.
    Returns an empty dictionary if the file doesn't exist or fails to load.
    """
    ensure_temp_directory_exists()
    if not os.path.isfile(THREADS_INFO_FILE):
        return {}
    try:
        with open(THREADS_INFO_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_chat_threads_info(chat_threads_info: Dict[str, Any]) -> None:
    """
    Saves the dictionary containing basic info for all threads to a JSON file.
    Called after any update to the metadata in session.
    """
    ensure_temp_directory_exists()
    with open(THREADS_INFO_FILE, "w", encoding="utf-8") as f:
        json.dump(chat_threads_info, f, indent=4)

def format_exchange(message):
    return f"User: {message['user']} \nAssistant: {message['answer']}\n"

def get_chat_history(chat_history):
    return [
        format_exchange(message)
        for message in chat_history['messages']
    ]

def cleanup_temp_dir():
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
        print(f"Deleted temp directory: {TEMP_DIR}")

# def export_chat_history(chat_text):
#     # Save chat history to a text file
#     with open("chat_history.txt", "w+") as f:
#         f.write(f"Chat History:\n{chat_text}")
#     print("Chat history exported to chat_history.txt")


# -------------------- Single Thread Data Management --------------------
# def get_thread_file_path(thread_id: str) -> str:
#     """
#     Returns the file path where a specific thread's data (messages, embeddings, etc.) will be stored.
#     """
#     ensure_temp_directory_exists()
#     return os.path.join(TEMP_DIR, f"thread_{thread_id}.json")


# def load_thread_data(thread_id: str) -> Dict[str, Any]:
#     """
#     Loads the thread data (messages, embeddings, etc.) from the corresponding JSON file.
#     Returns an empty dict if the file doesn't exist or fails to load.
#     """
#     file_path = get_thread_file_path(thread_id)
#     if not os.path.isfile(file_path):
#         return {}
#     try:
#         with open(file_path, "r", encoding="utf-8") as f:
#             return json.load(f)
#     except (json.JSONDecodeError, IOError):
#         return {}


# def save_thread_data(thread_id: str, thread_data: Dict[str, Any]) -> None:
#     """
#     Persists the thread data into a separate JSON file for that thread.
#     Called after new messages or other data are added to the thread.
#     """
#     file_path = get_thread_file_path(thread_id)
#     ensure_temp_directory_exists()
#     with open(file_path, "w", encoding="utf-8") as f:
#         json.dump(thread_data, f, indent=4)
