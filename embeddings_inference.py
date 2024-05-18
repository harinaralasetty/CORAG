
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

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
	
	return generate(prompt)

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
	similarities = cosine_similarity(embeddings, query_vector)[0]
	highest_scored_index = similarities.argmax()
	return highest_scored_index

def fetch_relevant_data(query_vector, embeddings, original_texts):
	most_relevant_index = find_closest_embeddings(query_vector, embeddings)
	return original_texts[most_relevant_index]

