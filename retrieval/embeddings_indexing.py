from math import ceil
from time import time

import hnswlib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import config
from retrieval.embeddings_router import get_embeddings_model


def initialize_hnsw_indexing(vectors):
    vectors_np = np.array(vectors, dtype=np.float32)
    if vectors_np.ndim != 2:
        raise ValueError(f"Expected 2D array for vectors, got shape {vectors_np.shape}")

    index = hnswlib.Index(space='cosine', dim=vectors_np.shape[1])
    index.init_index(
        max_elements=10000,
        ef_construction=config.HNSW_CONFIG["ef_construction"],
        M=config.HNSW_CONFIG["M"],
    )
    index.add_items(vectors_np)
    index.set_ef(50)
    return index


def _chunk_overlapping(text, chunk_size, overlap):
    step = chunk_size - overlap
    return [text[i:i + chunk_size] for i in range(0, len(text) - chunk_size + 1, step)]


def generate_embeddings(extracted_text, vectors=None, embedding_provider="google"):
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

    if not vectors:
        vectors = initialize_hnsw_indexing(vector_lst)
    else:
        vectors.add_items(np.array(vector_lst))

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


def find_closest_embeddings_hnsw(query_vector, embeddings):
    k = min(20, len(embeddings.get_ids_list()))
    labels, _ = embeddings.knn_query(query_vector, k=k)
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

    most_relevant_indexes = find_closest_embeddings_hnsw(query_vector, embeddings)
    candidate_vectors = [embeddings.get_items([i])[0] for i in most_relevant_indexes]
    reranked_indexes = rerank_results(query_vector, candidate_vectors, most_relevant_indexes)
    return "".join(original_texts[i] for i in reranked_indexes)
