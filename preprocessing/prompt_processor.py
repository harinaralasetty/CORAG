from config import QUERY_REPHRASAL_PROMPT, CHAT_NAMING_PROMPT
from inference.gemini_interaction import generate_gemini_response

def rephrase_prompt(prompt, document_theme, model):
	rephrasal_request = QUERY_REPHRASAL_PROMPT.format(prompt = prompt, document_theme = document_theme)
	print(f"```\n{rephrasal_request}\n```")

	rephrased_prompt = generate_gemini_response(rephrasal_request, model)
	print("REPHRASED", rephrased_prompt)

	return rephrased_prompt

def chat_namer(request, response, model):
	chat_naming_request = CHAT_NAMING_PROMPT.format(request, response)
	print(f"```\n{chat_naming_request}\n```")

	chat_name = generate_gemini_response(chat_naming_request, model)
	print("CHAT NAME", chat_name)

	return chat_name
	   

