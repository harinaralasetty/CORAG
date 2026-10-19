from inference.gemini_interaction import generate_gemini_response
from inference.claude_interaction import generate_claude_response
from inference.ollama_interaction import OLLAMA_PREFIX, generate_ollama_response


def is_claude(model):
    return bool(model) and model.startswith("claude")


def is_ollama(model):
    return bool(model) and model.startswith(OLLAMA_PREFIX)


def uses_tools(model):
    """Only Gemini models go through the LangChain tool chain; the others get a plain prompt."""
    return not is_claude(model) and not is_ollama(model)


def generate_response(prompt, model):
    if is_claude(model):
        return generate_claude_response(prompt, model)
    if is_ollama(model):
        return generate_ollama_response(prompt, model)
    return generate_gemini_response(prompt, model)
