from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-pro")

def generate_gemini_response(prompt, model):
    """
    Directly interacts with the Gemini model to generate a response.
    """
    try:
        result = llm.invoke(prompt)
        return result.content
    except Exception as e:
        print(f"Error in Gemini interaction: {e}")
        return "Sorry, I couldn't process your request."

def get_model_instance(model):
    llm = ChatGoogleGenerativeAI(model=model)
    return llm 

# from streamlit_callback import StreamlitCallbackHandler
# from langchain.callbacks.manager import CallbackManager
# from langchain_google_genai import ChatGoogleGenerativeAI
# import streamlit as st

# def get_streaming_llm():
#     # Create a Streamlit placeholder to show streamed output
#     placeholder = st.empty()
#     streamlit_handler = StreamlitCallbackHandler(placeholder)

#     callback_manager = CallbackManager([streamlit_handler])
    
#     # Enable streaming on Gemini
#     llm = ChatGoogleGenerativeAI(
#         model="gemini-pro",
#         streaming=True,
#         callback_manager=callback_manager
#     )
#     return llm

# llm = get_streaming_llm()