
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 

def retrieve_answer_from_gemini(PROMPT, question, original_data, vectors, chat_history_sentences, chat_history_vectors):
	query_vector = generate_embeddings([question])
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
	print(f"\n\nPrompt:'''{prompt}'''")
	
	response = generate(prompt)
	print(f"\nGemini response:{response}")
	
	return response

def generate(prompt):
	llm = ChatGoogleGenerativeAI(model="gemini-pro")
	result = llm.invoke(prompt)
	return result.content

def generate_embeddings(texts):
	embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
	vector = embeddings.embed_documents(texts)
	return vector

def generate_query_embedding(query_text):
	embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
	query_vector = embeddings.embed_query(text= query_text) 
	return query_vector

def find_closest_embeddings(query_vector, embeddings):
    similarities = cosine_similarity(embeddings, query_vector)
    similarities = cosine_similarity(embeddings, query_vector)

    sorted_data = np.column_stack((similarities, np.arange(len(similarities))))  
    sorted_data = sorted_data[sorted_data[:, 0].argsort()[::-1]] 

    top_5_indices = sorted_data[:5, 1].astype(int) 
    return top_5_indices

def fetch_relevant_data(query_vector, embeddings, original_texts):
	most_relevant_indexes = find_closest_embeddings(query_vector, embeddings)
	relevant_context = "".join([original_texts[x] for x in most_relevant_indexes])
	print(f"\n\nRelevant Context:'{relevant_context}'")
	return relevant_context

