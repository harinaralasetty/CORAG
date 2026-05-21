from config import QUERY_REPHRASAL_PROMPT, CHAT_NAMING_PROMPT
from inference.model_router import generate_response


def rephrase_prompt(prompt, document_theme, model):
    rephrasal_request = QUERY_REPHRASAL_PROMPT.format(prompt=prompt, document_theme=document_theme)
    return generate_response(rephrasal_request, model)


def chat_namer(request, response, model):
    chat_naming_request = CHAT_NAMING_PROMPT.format(request, response)
    return generate_response(chat_naming_request, model)
