"""The error raised when a call to a model fails."""
import os

KEY_VARIABLES = ("GOOGLE_API_KEY", "ANTHROPIC_API_KEY")


class LLMError(RuntimeError):
    """A model call failed. The message names the cause with any API key removed, so it can be shown."""

    def __init__(self, error):
        message = f"{type(error).__name__}: {error}"
        for name in KEY_VARIABLES:
            key = os.environ.get(name)
            if key:
                message = message.replace(key, "[redacted]")
        super().__init__(f"The model request failed ({message})")
