# retrieval/embeddings_indexing.py

import config
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
from time import time 
import hnswlib
from math import ceil

# Initialize the embeddings model
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

import numpy as np
import hnswlib

def initialize_hnsw_indexing(vectors):
    # Convert the list to a NumPy array
    vectors_np = np.array(vectors, dtype=np.float32)  # Ensure numeric type

    # Ensure vectors_np is 2D
    if len(vectors_np.shape) != 2:
        raise ValueError(f"Expected 2D array for vectors, but got shape {vectors_np.shape}")

    # Get the dimension from the array shape
    dimension = vectors_np.shape[1]

    # Create a new index
    hnswlib_indexing = hnswlib.Index(space='cosine', dim=dimension)
    hnswlib_indexing.init_index(
        max_elements=10000, 
        ef_construction=config.HNSW_CONFIG["ef_construction"], 
        M=config.HNSW_CONFIG["M"]
    )

    # Add vectors to the index
    hnswlib_indexing.add_items(vectors_np)  # Correctly formatted

    hnswlib_indexing.set_ef(50)

    print("HNSW index initialized successfully.")  # Debugging

    return hnswlib_indexing

def generate_embeddings(extracted_text, vectors=None):
    if vectors is None:
        vectors = []

    # Retrieve chunking settings from config
    chunk_size = config.CHUNK_SETTINGS["CHUNK_SIZE"]
    chunk_overlap = config.CHUNK_SETTINGS["CHUNK_OVERLAP"]

    # Validate chunking settings
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size for proper chunking.")

    # Function to chunk text
    def chunk_by_length(text, chunk_size):
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i + chunk_size])
        return chunks

    # Generate non-overlapping chunks
    chunks = chunk_by_length(extracted_text, chunk_size)

    # Create overlapping chunks
    overlapping_texts = []
    for i in range(0, len(extracted_text) - chunk_size + 1, chunk_size - chunk_overlap):
        chunk = extracted_text[i:i + chunk_size]
        overlapping_texts.append(chunk)

    # Ensure overlapping_texts is not empty
    if not overlapping_texts:
        raise ValueError("No chunks were generated. Check chunk_size and chunk_overlap settings.")

    # Use overlapping_texts as the final processed text
    texts = overlapping_texts

    texts = overlapping_texts
    
    print(f"\nGenerating embeddings for {len(texts)} chunks.")
    start_time = time()

    vector_lst = []
    batch_size = 100
    total_batches = ceil(len(texts)/batch_size)
    for x, batch_start in enumerate(range(0, len(texts), batch_size)):
        batch_end = min(batch_start + batch_size, len(texts))
        print(f"Embedding batch {x+1}/{total_batches}: sentences {batch_start} to {batch_end}")

        batch = texts[batch_start: batch_end]
        generated_embeddings = embeddings_model.embed_documents(batch)

        # Ensure generated_embeddings is a list of lists/arrays
        if isinstance(generated_embeddings, (list, tuple)):
            # Handle single-vector case
            if all(isinstance(vec, (float, int)) for vec in generated_embeddings):
                # Single vector returned as a flat list
                generated_embeddings = [generated_embeddings]
            vector_lst.extend(generated_embeddings)
        else:
            raise ValueError("embed_documents did not return a list or tuple of embeddings.")

    end_time = time()

    print(f"\nGenerated embeddings in {end_time - start_time} seconds.")
    print(f"Total Embeddings Generated: {len(vector_lst)}")  # Debugging

    # Add vectors to HNSW index 
    if not vectors:
        vectors = initialize_hnsw_indexing(vector_lst)
    else:
        vectors.add_items(np.array(vector_lst))

    return vectors, texts

def generate_query_embedding(query_text):
    query_vector = embeddings_model.embed_documents([query_text]) 
    # Ensure query_vector is 2D
    if isinstance(query_vector, list):
        if all(isinstance(vec, (float, int)) for vec in query_vector):
            query_vector = [query_vector]
    elif isinstance(query_vector, np.ndarray):
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
    else:
        raise ValueError("embed_documents returned an unsupported type for query embedding.")
    return query_vector

def find_closest_embeddings_hnsw(query_vector, embeddings):
    start_time = time()
    labels, distances = embeddings.knn_query(query_vector, k=min(20, len(embeddings.get_ids_list())))
    end_time = time()

    return labels[0]

def rerank_results(query_vector, candidate_vectors, candidate_indexes):
    # Calculate cosine similarity for re-ranking
    similarity_scores = cosine_similarity(query_vector, candidate_vectors)[0]  
    ranked_indexes = np.argsort(similarity_scores)[::-1]  
    return [candidate_indexes[i] for i in ranked_indexes]  

def fetch_relevant_data(query_vector, embeddings, original_texts):
    if not isinstance(embeddings, hnswlib.Index):
        raise ValueError("The 'embeddings' parameter must be an HNSW index object.")

    # Find the most relevant embeddings using the HNSW index
    most_relevant_indexes = find_closest_embeddings_hnsw(query_vector, embeddings)

    # Retrieve candidate vectors and rerank
    candidate_vectors = [embeddings.get_items([index])[0] for index in most_relevant_indexes]
    reranked_indexes = rerank_results(query_vector, candidate_vectors, most_relevant_indexes)

    # Fetch the relevant context based on reranked indexes
    relevant_context = "".join([original_texts[x] for x in reranked_indexes]) if len(original_texts) > 0 else ""

    # print(f"Relevant context: {relevant_context}")
    return relevant_context
