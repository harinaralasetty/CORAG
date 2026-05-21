from anthropic import Anthropic

client = Anthropic()

DEFAULT_CLAUDE_MODEL = "claude-haiku-4-5-20251001"

def generate_claude_response(prompt, model=None):
    try:
        msg = client.messages.create(
            model=model or DEFAULT_CLAUDE_MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except Exception as e:
        print(f"Error in Claude interaction: {e}")
        return "Sorry, I couldn't process your request."
