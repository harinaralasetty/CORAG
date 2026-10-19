from functools import lru_cache

from openai import OpenAI

from config import OLLAMA_BASE_URL, OLLAMA_PREFIX
from inference.llm_error import LLMError


@lru_cache(maxsize=None)
def get_client():
    # Ollama serves an OpenAI-compatible API; it ignores the key, but the client requires one.
    # No retries: a local server that is down or timed out will not do better on a second try.
    return OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama", max_retries=0)


def generate_ollama_response(prompt, model):
    try:
        reply = get_client().chat.completions.create(
            model=model.removeprefix(OLLAMA_PREFIX),
            messages=[{"role": "user", "content": prompt}],
        )
        return reply.choices[0].message.content
    except Exception as e:
        raise LLMError(e) from e
