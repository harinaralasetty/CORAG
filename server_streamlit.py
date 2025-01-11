import streamlit as st
from inference.inference_manager import process_answer, process_embeddings
from chat_management.chat_utils import load_chat_threads_info, save_chat_threads_info, get_chat_history, cleanup_temp_dir
from preprocessing.pdf_processor import process_pdf
from preprocessing.audio_processor import transcribe_audio
from preprocessing.prompt_processor import chat_namer
import traceback
import config
import uuid
import base64
from datetime import datetime
from cosmetics import apply_cosmetics, get_random_greeting
import atexit

# Register cleanup function to remove temp file when server is terminated.
atexit.register(cleanup_temp_dir)

# --------- Streamlit Page Configuration ---------
st.set_page_config(
    page_title="CORAG",
    page_icon="CORAG_ICON.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

hide_decoration_bar_style = '''
    <style>
        header {visibility: hidden;}
    </style>
'''
st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)

# --------- Custom CSS for Better Styling ---------
apply_cosmetics()

# --------- Initialize Session States ---------
if 'chat_threads' not in st.session_state:
    st.session_state.chat_threads = load_chat_threads_info()

if 'current_thread' not in st.session_state:
    st.session_state.current_thread = None

if 'vectors_per_thread' not in st.session_state:
    # { thread_id: <list of embeddings> }
    st.session_state.vectors_per_thread = {}

if 'text_per_thread' not in st.session_state:
    # { thread_id: <list of text segments> }
    st.session_state.text_per_thread = {}

if 'new_thread_pending' not in st.session_state:
    st.session_state.new_thread_pending = False

if 'thread_temp_data' not in st.session_state:
    st.session_state.thread_temp_data = None


# --------- Caching for PDF/Audio processing ---------
@st.cache_data
def cached_process_pdf(file):
    return process_pdf(file)

@st.cache_data
def cached_transcribe_audio(file):
    return transcribe_audio(file)


# --------- Sidebar ---------
with st.sidebar:
    try:
        icon_base64 = base64.b64encode(open("CORAG_ICON.png", "rb").read()).decode("utf-8")
    except FileNotFoundError:
        icon_base64 = ""

    # Title with an icon
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 1rem;">
            <img src="data:image/png;base64,{icon_base64}" alt="App Icon" 
                style="width: 3rem; height: 3rem; margin-bottom: -0.3rem;">
            <h1 style="font-size: 3rem; font-weight: bold; margin: 0;">
                CORAG
            </h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    
    # Model selection
    st.title("Models")
    selected_model = st.selectbox("Current", config.models_list , key="selected_model")
    st.divider()

    # Chats section
    st.title("Chats")
    
    # + New Thread button
    if st.button("\\+ Create New Thread", key="new_thread_button", use_container_width=True):
        st.session_state.new_thread_pending = True
        st.session_state.current_thread = None
        st.rerun()

    # Sort threads by last_updated_at desc
    sorted_threads = sorted(
        st.session_state.chat_threads.items(),
        key=lambda item: item[1].get('last_updated_at', item[1].get('created_at', '')),
        reverse=True
    )

    # Display existing threads
    for thread_id, thread_data in sorted_threads:
        thread_name = thread_data.get("name", "Unnamed Thread")
        is_selected = (thread_id == st.session_state.current_thread)

        if is_selected:
            st.markdown(
                f"<div class='thread-box selected'>{thread_name}</div>",
                unsafe_allow_html=True
            )
        else:
            if st.button(thread_name, key=f"thread_btn_{thread_id}", use_container_width=True):
                st.session_state.current_thread = thread_id
                st.session_state.new_thread_pending = False
                st.rerun()


# --------- Main Content Area ---------
if st.session_state.current_thread:
    current_thread_data = st.session_state.chat_threads[st.session_state.current_thread]
    current_thread_name = current_thread_data.get("name", "Unnamed Thread")
    st.title(f"{current_thread_name}")
    
    # Display the uploaded files for this thread
    processed_files = current_thread_data.get('processed_files', [])
    if processed_files:
        st.markdown("#### Files uploaded in this thread:")
        for file_info in processed_files:
            st.write(f"- **{file_info['name']}** (type: {file_info['type']})")
else:
    st.title("Hey there!")


# --------- File Uploader and Controls ---------
uploader_key = (
    f"uploader_{st.session_state.current_thread}" 
    if st.session_state.current_thread 
    else "uploader_none"
)
with st.container():
    uploaded_file = st.file_uploader(
        "📄 Upload Documents (PDF, MP3, WAV)",
        type=["pdf", "mp3", "wav"],
        help="Upload your documents here for analysis",
        key=uploader_key
    )


# --------- Chat Display ---------
st.divider()
with st.container():
    if st.session_state.current_thread:
        messages = current_thread_data.get("messages", [])
        if len(messages) == 0:
            st.info(get_random_greeting())
        else:
            for message in messages:
                if message.get('user'):
                    with st.chat_message("user", avatar="👤"):
                        st.markdown(message['user'])
                if message.get('answer'):
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(message['answer'])
    else:
        st.info(get_random_greeting())


# --------- User Input for Chat ---------
user_question = st.chat_input("Type your question here...")


# --------- Handle Chat Submission ---------
if user_question:
    # If no thread or user clicked +New Thread, create one
    if st.session_state.new_thread_pending or not st.session_state.current_thread:
        thread_id = str(uuid.uuid4())
        st.session_state.current_thread = thread_id
        st.session_state.thread_temp_data = {
            "name": "Temporary Thread",
            "messages": [],
            "processed_files": [],
            "document_theme": "",
            "last_updated_at": datetime.now().isoformat(),
            "created_at": datetime.now().isoformat()
        }
        st.session_state.chat_threads[thread_id] = st.session_state.thread_temp_data
        st.session_state.new_thread_pending = False

    try:
        with st.spinner("Processing your query..."):
            # Retrieve the thread object
            thread_id = st.session_state.current_thread
            current_thread = st.session_state.chat_threads[thread_id]

            # Check for newly uploaded file
            if uploaded_file is not None:
                # Only process if new
                already_processed = any(
                    f["name"] == uploaded_file.name
                    for f in current_thread["processed_files"]
                )
                if not already_processed:
                    with st.spinner("Processing file..."):
                        if uploaded_file.type == "application/pdf":
                            extracted_text, theme = cached_process_pdf(uploaded_file)

                        elif uploaded_file.type in ["audio/mpeg", "audio/wav"]:
                            extracted_text, theme = cached_transcribe_audio(uploaded_file)
                        else:
                            st.warning("Unsupported file type uploaded.")
                            extracted_text, theme = "", ""

                        if extracted_text:
                            # Save info about the file
                            current_thread["processed_files"].append(
                                {"name": uploaded_file.name, "type": uploaded_file.type}
                            )
                            # Store theme
                            current_thread["document_theme"] = theme

                            # Generate embeddings in inference_manager
                            vectors, original_data = process_embeddings(extracted_text)

                            # Save both to session_state
                            st.session_state.text_per_thread[thread_id] = original_data
                            st.session_state.vectors_per_thread[thread_id] = vectors
                        else:
                            st.warning("No text extracted from file.")
                # If already processed, do nothing special here

            # Grab current known vectors and texts
            vectors = st.session_state.vectors_per_thread.get(thread_id, [])
            original_data = st.session_state.text_per_thread.get(thread_id, [])

            # Now pass them to process_answer
            answer = process_answer(
                config.BASE_PROMPT,
                thread_name=thread_id,
                question=user_question,
                original_data=original_data,
                vectors=vectors,
                chat_history=get_chat_history(current_thread),
                document_theme=current_thread.get("document_theme", ""), 
                inference_model = selected_model
            )

            # Store the new Q&A
            current_thread.setdefault("messages", []).append(
                {"user": user_question, "answer": answer}
            )

            # Rename thread if it's still "Temporary Thread"
            if current_thread.get("name") == "Temporary Thread":
                new_thread_name = chat_namer(user_question, answer, selected_model)
                current_thread["name"] = new_thread_name

            # Update last_updated_at
            current_thread["last_updated_at"] = datetime.now().isoformat()
            save_chat_threads_info(st.session_state.chat_threads)

            st.rerun()  # Refresh the UI

    except Exception as e:
        st.error(f"Error: {e}")
        st.error(traceback.format_exc())
