from functools import lru_cache
from langchain_google_genai import ChatGoogleGenerativeAI


@lru_cache(maxsize=None)
def get_model_instance(model):
    return ChatGoogleGenerativeAI(model=model)


def generate_gemini_response(prompt, model):
    try:
        return get_model_instance(model).invoke(prompt).content
    except Exception as e:
        print(f"Error in Gemini interaction: {e}")
        return "Sorry, I couldn't process your request."
