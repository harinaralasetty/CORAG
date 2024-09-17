import gradio as gr
import os 

from embeddings_indexing import generate_embeddings
from inference import retrieve_answer_from_gemini
from chat_utils import export_chat_history, get_chat_history, get_clean_chat_history, format_exchange
from pdf_processor import process_pdf

import config

# from chunking import initiate_chunking

CWD = os.getcwd()


# global variables 
chat_history = {'messages': []}
chat_history_vectors = []

original_data = []
vectors = []

pdf_processed = False 
processed_pdf_name = ""

# Export chat option (optional)
export_button = gr.Button(value="Export Chat")
export_button.click = lambda: export_chat_history("")

async def rag_application_function(uploaded_pdf, question):
    # use global variables 
    global chat_history, chat_history_vectors, export_button, vectors, original_data, pdf_processed, processed_pdf_name

    # process the uploaded file
    if uploaded_pdf and pdf_processed == False and uploaded_pdf.name!=processed_pdf_name: 
        try:
            extracted_text = process_pdf(uploaded_pdf)
            pdf_processed = True 
            processed_pdf_name = uploaded_pdf.name 

        except Exception as e:
            print(f"Error extracting text: {e}")
            extracted_text = "Error: Could not extract text from PDF."
        
        finally: 
            context = extracted_text
            original_data = context.split('.')
            vectors = generate_embeddings(original_data, vectors = vectors) 

    # inference
    message_id = len(chat_history['messages'])  
    new_exchange = {
        'id': message_id,
        'user': question,
        'answer': retrieve_answer_from_gemini(
            config.PROMPT, 
            question, 
            original_data, 
            vectors, 
            get_chat_history(chat_history),
            chat_history_vectors,
            )
    }
    chat_history['messages'].append(new_exchange)

    # Format chat history for display
    chat_text = get_clean_chat_history(chat_history)

    # Call export_chat_history directly with chat_history
    export_chat_history(chat_text)

    # add new chat to chat embeddings 
    chat_history_vectors = generate_embeddings([format_exchange(new_exchange)], chat_history_vectors)

    # Return the chat history and the download link
    return chat_text, gr.Button("Export Chat", link=f'/file={CWD}/chat_history.txt')

interface = gr.Interface(
    fn=rag_application_function,
    inputs=[gr.File(label="Upload PDF Document"), gr.Textbox(label="Ask a Question")],
    outputs=[gr.Text(label="Chat History"), gr.Button(value="Export Chat")],
    title= config.APP_TITLE,
    description= config.DESCRIPTION
)

interface.launch(allowed_paths=[f"{CWD}/"])
