# unused at the moment
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler

class StreamlitCallbackHandler(BaseCallbackHandler):
    """
    A simple callback handler to stream tokens in real-time to Streamlit.
    """
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.accumulated_text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.accumulated_text += token
        # Update placeholder text in real-time
        self.placeholder.markdown(self.accumulated_text)

