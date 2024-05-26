
from langchain_google_genai import ChatGoogleGenerativeAI
from embeddings_indexing import generate_query_embedding, fetch_relevant_data

def generate_gemini_response(prompt):
	llm = ChatGoogleGenerativeAI(model="gemini-pro")
	result = llm.invoke(prompt)
	return result.content

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
	
	response = generate_gemini_response(prompt)
	print(f"\nGemini response: {response}")
	return response