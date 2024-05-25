
import config
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
from time import time 
import hnswlib
import os 
from math import ceil

SIMILARITY_MODE = os.environ.get("SIMILARITY_MODE")
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def retrieve_answer_from_gemini(PROMPT, question, original_data, vectors, chat_history_sentences, chat_history_vectors):
	query_vector = generate_query_embedding(question)
	relevant_context = ""
	relevant_chat_history = ""

	# fetch relevant contextual data if exists 
	if len(original_data):
		relevant_context = fetch_relevant_data(query_vector, vectors, original_data)
	
	# fetch relevant chat history if exists
	if len(chat_history_sentences):
		relevant_chat_history = fetch_relevant_data(query_vector, chat_history_vectors, chat_history_sentences)
    
	prompt = PROMPT.format(
		context = relevant_context, 
		question = question, 
		relevant_chat_history = relevant_chat_history)

	print(f"\n\nPrompt:'''\n{prompt}\n'''")
	
	response = generate(prompt)
	print(f"\nGemini response: {response}")
	
	return response

def generate(prompt):
	llm = ChatGoogleGenerativeAI(model="gemini-pro")
	result = llm.invoke(prompt)
	return result.content

def initialize_hnsw_indexing(vectors):

    # Assuming vectors is a list of lists, where each inner list represents a vector
    # Calculate the dimensionality by finding the maximum length of the vectors
	global hnswlib_indexing

	# Convert the list to a NumPy array
	vectors_np = np.array(vectors)

	# Get the dimension from the array shape
	dimension = vectors_np.shape[1]

	# Create a new index
	hnswlib_indexing = hnswlib.Index(space='cosine', dim=dimension)
	hnswlib_indexing.init_index(max_elements=10000, ef_construction=200, M=16)

	# Add each vector as a single element array
	for vector in vectors:
		hnswlib_indexing.add_items(np.array([vector]))  # Wrap in a single-element array

	hnswlib_indexing.set_ef(50)

	return hnswlib_indexing

def generate_embeddings(texts, vectors = []):
	global SIMILARITY_MODE, embeddings_model

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

	# adding vectors to hnswlib index if similarity mode is hnsw
	if SIMILARITY_MODE == "cosine":
		vectors.extend(vector_lst)
	if SIMILARITY_MODE == "hnsw":
		# initialize the first time 
		if vectors == []:
			vectors = initialize_hnsw_indexing(vector_lst)
		else: 
			vectors.add_items(vector_lst)

	return vectors

def generate_query_embedding(query_text):
	global embeddings_model
	query_vector = embeddings_model.embed_documents([query_text]) 
	return query_vector

def find_closest_embeddings(query_vector, embeddings):
	# for normal cosine 
	print(f"Finding close matches for query among {len(embeddings)} embeddings.")
	start_time = time()

	similarities = cosine_similarity(embeddings, query_vector)
	end_time = time()

	print(f"\n\nGenerated close matches in {end_time - start_time} seconds in '{SIMILARITY_MODE}' mode..")

	sorted_data = np.column_stack((similarities, np.arange(len(similarities))))  
	sorted_data = sorted_data[sorted_data[:, 0].argsort()[::-1]] 

	top_5_indices = sorted_data[:5, 1].astype(int) 
	return top_5_indices

def find_closest_embeddings_hnsw(query_vector, embeddings):
	# for hnsl
	start_time = time()
	labels, distances = embeddings.knn_query(query_vector, k= min(5, len(embeddings.get_ids_list()) ))
	end_time = time()
	print(f"\n\nGenerated close matches in {end_time - start_time} seconds in '{SIMILARITY_MODE}' mode..")
	return labels[0]

def fetch_relevant_data(query_vector, embeddings, original_texts):
	global SIMILARITY_MODE

	if SIMILARITY_MODE == "cosine":
		most_relevant_indexes = find_closest_embeddings(query_vector, embeddings)
	elif SIMILARITY_MODE == "hnsw":
		most_relevant_indexes = find_closest_embeddings_hnsw(query_vector, embeddings)
	
	relevant_context = "".join([original_texts[x] for x in most_relevant_indexes])
	print(f"\n\nRelevant Context:'{relevant_context}'")
	return relevant_context

