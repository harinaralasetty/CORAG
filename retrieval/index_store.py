"""Save and load a chat's vector index and chunk texts so documents survive restarts."""
import json
import os

import hnswlib

INDEX_FILE = "index.bin"
CHUNKS_FILE = "chunks.json"
MIN_CAPACITY = 1024
EF_SEARCH = 50


def save_thread_index(base_dir, thread_id, index, texts):
    folder = os.path.join(base_dir, thread_id)
    os.makedirs(folder, exist_ok=True)
    index.save_index(os.path.join(folder, INDEX_FILE))
    with open(os.path.join(folder, CHUNKS_FILE), "w", encoding="utf-8") as f:
        json.dump({"dim": index.dim, "texts": texts}, f)


def load_thread_index(base_dir, thread_id):
    """Return (index, texts), or (None, []) when nothing valid is stored."""
    folder = os.path.join(base_dir, thread_id)
    index_path = os.path.join(folder, INDEX_FILE)
    chunks_path = os.path.join(folder, CHUNKS_FILE)
    if not (os.path.isfile(index_path) and os.path.isfile(chunks_path)):
        return None, []
    try:
        with open(chunks_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        texts = meta["texts"]
        index = hnswlib.Index(space="cosine", dim=meta["dim"])
        index.load_index(index_path, max_elements=max(len(texts) * 2, MIN_CAPACITY))
        index.set_ef(EF_SEARCH)
    except Exception:
        return None, []
    if index.get_current_count() != len(texts):
        return None, []
    return index, texts
