
import config
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
from time import time 
import hnswlib
import os 
from math import ceil


embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def initialize_hnsw_indexing(vectors):

	global hnswlib_indexing

	# Convert the list to a NumPy array
	vectors_np = np.array(vectors)

	# Get the dimension from the array shape
	dimension = vectors_np.shape[1]

	# Create a new index
	hnswlib_indexing = hnswlib.Index(space='cosine', dim=dimension)
	hnswlib_indexing.init_index(
		max_elements=10000, 
		ef_construction=config.HNSW_CONFIG["ef_construction"], 
		M=config.HNSW_CONFIG["M"])

	# Add each vector as a single element array
	for vector in vectors:
		hnswlib_indexing.add_items(np.array([vector]))  # Wrap in a single-element array

	hnswlib_indexing.set_ef(50)

	return hnswlib_indexing

def generate_embeddings(texts, vectors = [], ):

	# Retrieve chunking settings from config
	chunk_size = min(config.CHUNK_SETTINGS["CHUNK_SIZE"], len(texts))
	chunk_overlap = min(config.CHUNK_SETTINGS["CHUNK_OVERLAP"], len(texts)-1)

	# Validate chunking settings
	if chunk_overlap > chunk_size:
		raise ValueError("chunk_overlap must be smaller than chunk_size for proper chunking.")

	overlapping_texts = []

	# Iterate through the text with a step size of chunk_size - chunk_overlap to create overlapping chunks
	for i in range(0, len(texts) - chunk_size + 1, chunk_size - chunk_overlap):
		chunk = ' '.join(texts[i:i + chunk_size])
		overlapping_texts.append(chunk)

	# If no chunks were generated (possible when chunk_size is too large)
	if not overlapping_texts:
		raise ValueError("No chunks were generated. Check chunk_size and chunk_overlap settings.")

	texts = overlapping_texts
	
	print(f"\nGenerating embeddings for {len(texts)} sentences.")
	start_time = time()

	vector_lst = []
	batch_size = 100
	total_batches = ceil( len(texts)/batch_size )
	for x, batch_start in enumerate(range(0, len(texts), batch_size)) :
		batch_end = min(batch_start + batch_size, len(texts))
		print(f"Embedding batch {x+1}/{total_batches}: sentences {batch_start} to {batch_end}")

		batch = texts[ batch_start:  batch_end]
		generated_embeddings = embeddings_model.embed_documents(batch)
		vector_lst.extend(generated_embeddings)

	end_time = time()

	print(f"\nGenerated embeddings in {end_time-start_time} seconds.")

	# adding vectors to hnswlib index 
	if vectors == []: vectors = initialize_hnsw_indexing(vector_lst);
	else: vectors.add_items(vector_lst);

	return vectors #/ np.linalg.norm(vectors, axis=1, keepdims=True)

def generate_query_embedding(query_text):
	global embeddings_model
	query_vector = embeddings_model.embed_documents([query_text]) 
	return query_vector

def find_closest_embeddings_hnsw(query_vector, embeddings):
	# for hnsl
	start_time = time()
	labels, distances = embeddings.knn_query(query_vector, k= min(5, len(embeddings.get_ids_list()) ))
	end_time = time()
	print(f"\n\nFound close matches in {end_time - start_time} seconds...")
	return labels[0]

def fetch_relevant_data(query_vector, embeddings, original_texts):

	most_relevant_indexes = find_closest_embeddings_hnsw(query_vector, embeddings)
	
	relevant_context = "".join([original_texts[x] for x in most_relevant_indexes])
	print(f"\n\nRelevant Context:'{relevant_context}'")
	return relevant_context

