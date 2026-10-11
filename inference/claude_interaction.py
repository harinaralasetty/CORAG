from functools import lru_cache

from anthropic import Anthropic

DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5-20251001"

@lru_cache(maxsize=None)
def get_client():
    return Anthropic()


def generate_claude_response(prompt, model=None):
    try:
        msg = get_client().messages.create(
            model=model or DEFAULT_CLAUDE_MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except Exception as e:
        print(f"Error in Claude interaction: {e}")
        return f"Sorry, I couldn't process your request ({type(e).__name__}: {e})"
