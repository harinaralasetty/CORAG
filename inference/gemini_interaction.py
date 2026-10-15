from functools import lru_cache
from langchain_google_genai import ChatGoogleGenerativeAI

from inference.llm_error import LLMError


@lru_cache(maxsize=None)
def get_model_instance(model):
    return ChatGoogleGenerativeAI(model=model)


def generate_gemini_response(prompt, model):
    try:
        return get_model_instance(model).invoke(prompt).content
    except Exception as e:
        raise LLMError(e) from e
