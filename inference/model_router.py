from inference.gemini_interaction import generate_gemini_response
from inference.claude_interaction import generate_claude_response


def is_claude(model):
    return bool(model) and model.startswith("claude")


def generate_response(prompt, model):
    if is_claude(model):
        return generate_claude_response(prompt, model)
    return generate_gemini_response(prompt, model)
