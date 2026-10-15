"""Save and load a chat's vector index and chunk texts so documents survive restarts."""
import json
import logging
import os

import hnswlib

from file_utils import fsync_file, make_temp_path, write_json_temp

INDEX_FILE = "index.bin"
CHUNKS_FILE = "chunks.json"
MIN_CAPACITY = 1024
EF_SEARCH = 50

log = logging.getLogger(__name__)


def save_thread_index(base_dir, thread_id, index, texts):
    folder = os.path.join(base_dir, thread_id)
    os.makedirs(folder, exist_ok=True)
    index_path = os.path.join(folder, INDEX_FILE)
    chunks_path = os.path.join(folder, CHUNKS_FILE)
    index_tmp = make_temp_path(index_path)
    try:
        index.save_index(index_tmp)
        fsync_file(index_tmp)
        chunks_tmp = write_json_temp(chunks_path, {"dim": index.dim, "texts": texts})
    except BaseException:
        os.remove(index_tmp)
        raise
    # Chunks go in last. A crash between the two renames leaves a new index with old chunks;
    # their counts differ (the index only grows), so load_thread_index rejects the pair.
    os.replace(index_tmp, index_path)
    os.replace(chunks_tmp, chunks_path)


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
        log.warning("Could not load the index of chat %s", thread_id, exc_info=True)
        return None, []
    if index.get_current_count() != len(texts):
        log.warning("Index of chat %s holds %d vectors but %d chunks", thread_id, index.get_current_count(), len(texts))
        return None, []
    return index, texts
