import gradio as gr
import os 

from embeddings_indexing import generate_embeddings
from inference import retrieve_answer_from_gemini
from chat_utils import export_chat_history, get_chat_history, get_clean_chat_history, format_exchange
from pdf_processor import process_pdf

from config import config 

CWD = os.getcwd()

SIMILARITY_MODE = os.environ.get("SIMILARITY_MODE")
print(f"Starting in '{SIMILARITY_MODE}' similarity mode...")

# read base prompt 
with open("base_prompt.txt", "r") as prompt_file:
    PROMPT = "".join( prompt_file.readlines() )

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
            PROMPT, 
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
    title= config["title"],
    description= config["description"]
)

interface.launch(allowed_paths=[f"{CWD}/"])
