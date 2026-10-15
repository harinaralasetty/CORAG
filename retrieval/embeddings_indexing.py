import threading
import weakref
from math import ceil
from time import time

import hnswlib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import config
from retrieval.embeddings_router import get_embeddings_model


MIN_CAPACITY = 1024
EF_SEARCH = 50

# hnswlib cannot resize or add to an index while it is being searched. An upload grows a chat's
# index on one worker thread while a question may search it on another, so each index gets a lock,
# held for one add (with its resize) or for one question's search and get_items.
_index_locks = weakref.WeakKeyDictionary()
_index_locks_guard = threading.Lock()


def index_lock(index):
    with _index_locks_guard:
        return _index_locks.setdefault(index, threading.Lock())


def initialize_hnsw_indexing(vectors):
    vectors_np = np.array(vectors, dtype=np.float32)
    if vectors_np.ndim != 2:
        raise ValueError(f"Expected 2D array for vectors, got shape {vectors_np.shape}")

    index = hnswlib.Index(space='cosine', dim=vectors_np.shape[1])
    index.init_index(
        max_elements=max(len(vectors_np), MIN_CAPACITY),
        ef_construction=config.HNSW_CONFIG["ef_construction"],
        M=config.HNSW_CONFIG["M"],
    )
    index.add_items(vectors_np)
    index.set_ef(EF_SEARCH)
    return index


def add_to_index(index, vectors):
    """Append vectors to an existing index, growing its capacity when needed."""
    vectors_np = np.array(vectors, dtype=np.float32)
    with index_lock(index):
        needed = index.get_current_count() + len(vectors_np)
        if needed > index.get_max_elements():
            index.resize_index(max(needed, index.get_max_elements() * 2))
        index.add_items(vectors_np)


def _chunk_overlapping(text, chunk_size, overlap):
    """Split text into overlapping chunks; the last chunk always reaches the end of the text."""
    if not text:
        return []
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(text), step):
        chunks.append(text[start:start + chunk_size])
        if start + chunk_size >= len(text):
            break
    return chunks


def generate_embeddings(extracted_text, vectors=None, embedding_provider="google", source=None):
    chunk_size = config.CHUNK_SETTINGS["CHUNK_SIZE"]
    chunk_overlap = config.CHUNK_SETTINGS["CHUNK_OVERLAP"]
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    texts = _chunk_overlapping(extracted_text, chunk_size, chunk_overlap)
    if not texts:
        raise ValueError("No chunks were generated. Check chunk_size/chunk_overlap settings.")

    embeddings_model = get_embeddings_model(embedding_provider)
    print(f"\nGenerating embeddings for {len(texts)} chunks via {embedding_provider}.")
    start_time = time()

    batch_size = 100
    total_batches = ceil(len(texts) / batch_size)
    vector_lst = []
    for i, batch_start in enumerate(range(0, len(texts), batch_size)):
        batch = texts[batch_start: batch_start + batch_size]
        print(f"Embedding batch {i + 1}/{total_batches} ({len(batch)} chunks)")
        generated = embeddings_model.embed_documents(batch)
        # Some providers return a flat list for a single-text input; normalize
        if generated and all(isinstance(v, (float, int)) for v in generated):
            generated = [generated]
        vector_lst.extend(generated)

    print(f"Generated {len(vector_lst)} embeddings in {time() - start_time:.2f}s")

    if vectors is None:
        vectors = initialize_hnsw_indexing(vector_lst)
    else:
        add_to_index(vectors, vector_lst)

    # The label is stored with the chunk (not embedded) so answers can cite the file
    if source:
        texts = [f"[Source: {source}]\n{t}" for t in texts]
    return vectors, texts


def generate_query_embedding(query_text, embedding_provider="google"):
    embeddings_model = get_embeddings_model(embedding_provider)
    query_vector = embeddings_model.embed_documents([query_text])
    # Normalize to 2D
    if isinstance(query_vector, list) and query_vector and all(isinstance(v, (float, int)) for v in query_vector):
        query_vector = [query_vector]
    elif isinstance(query_vector, np.ndarray) and query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)
    return query_vector


def find_closest_embeddings_hnsw(query_vector, embeddings, max_label=None):
    """Labels of the 20 nearest chunks; with max_label, only chunks whose label is below it are considered."""
    count = len(embeddings.get_ids_list())
    if max_label is None:
        labels, _ = embeddings.knn_query(query_vector, k=min(20, count))
        return labels[0]
    k = min(20, count, max_label)
    if k == 0:
        return []
    labels, _ = embeddings.knn_query(query_vector, k=k, filter=lambda label: label < max_label)
    return labels[0]


def rerank_results(query_vector, candidate_vectors, candidate_indexes):
    similarity_scores = cosine_similarity(query_vector, candidate_vectors)[0]
    ranked = np.argsort(similarity_scores)[::-1]
    return [candidate_indexes[i] for i in ranked]


def fetch_relevant_data(query_vector, embeddings, original_texts):
    if not isinstance(embeddings, hnswlib.Index):
        raise ValueError("'embeddings' must be an HNSW index object.")
    if not original_texts:
        return ""

    # An upload may grow the shared index while a question is answered: search only the chunks in original_texts
    with index_lock(embeddings):
        most_relevant_indexes = find_closest_embeddings_hnsw(query_vector, embeddings, max_label=len(original_texts))
        candidate_vectors = [embeddings.get_items([i])[0] for i in most_relevant_indexes]
    if len(most_relevant_indexes) == 0:
        return ""
    reranked_indexes = rerank_results(query_vector, candidate_vectors, most_relevant_indexes)
    return "\n\n".join(original_texts[i] for i in reranked_indexes)
