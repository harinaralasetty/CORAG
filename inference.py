
from langchain_google_genai import ChatGoogleGenerativeAI
from embeddings_indexing import generate_query_embedding, fetch_relevant_data
from toolkit import search, multiply
from config import PROMPT

from langchain_core.messages import AIMessage
from langchain_core.runnables import (
    Runnable,
)

tools = [multiply, search]
tool_map = {tool.name: tool for tool in tools}

llm = ChatGoogleGenerativeAI(model="gemini-pro")
llm_with_tools = llm.bind_tools(tools)

def call_tools(msg: AIMessage) -> Runnable:
	"""Simple sequential tool calling helper."""
	global tools, tool_map

	tool_calls = msg.tool_calls.copy()

	# If no tools are called, return the LLM response directly
	if not tool_calls:  
		return msg.content
	
	for tool_call in tool_calls:
		tool_call["output"] = tool_map[tool_call["name"]].invoke(tool_call["args"])

	tool_output = tool_call["output"]
	# new_prompt = f"The result of {tool_call}. Answer the query clearly based on the the tool's output. "
	print(f"PROMPT: \n{PROMPT}")
	new_prompt = PROMPT.format(
		context = "", 
		question = "", 
		relevant_chat_history = "",
		tool_call = tool_call)

	# Use LLM to generate a neat answer
	final_response = llm.invoke(new_prompt)
	return final_response
    
	# return tool_calls

def generate_gemini_response(prompt):
	global tools, tool_map, llm_with_tools

	try: 
		# if tool is needed to answer 
		chain = llm_with_tools | call_tools
		result = chain.invoke(prompt)

		# print("Gemini response:", result)
		# return result[0]['output']
		return result.content
	
	except Exception as e: 
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
		relevant_chat_history = relevant_chat_history,
		tool_call = "")

	print(f"\n\nPrompt:'''\n{prompt}\n'''")
	
	response = generate_gemini_response(prompt)
	print(f"\nGemini response: {response}")
	return response