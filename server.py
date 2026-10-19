"""CORAG — NiceGUI front-end.

Layout: header (logo + chat title), left drawer (models + chats), right drawer
(documents), main column (chat messages), footer (chat input).
"""
import asyncio
import io
import logging
import os
import tempfile
import traceback
import uuid
from datetime import datetime

from nicegui import ui

import config
from chat_management.chat_utils import (
    get_chat_history,
    load_chat_threads_info,
    save_chat_threads_info,
)
from cosmetics import get_random_greeting
from inference.inference_manager import process_answer, process_embeddings
from preprocessing.audio_processor import transcribe_audio
from preprocessing.pdf_processor import process_pdf
from preprocessing.text_processor import process_text
from preprocessing.prompt_processor import chat_namer
from retrieval.index_store import load_thread_index, save_thread_index


# ---------- App state (single-user local app) ----------
class State:
    chat_threads = load_chat_threads_info()
    current_thread_id: str | None = None
    vectors_per_thread: dict = {}
    text_per_thread: dict = {}
    uploads_in_progress: set = set()  # (chat id, file name) pairs still being processed
    selected_model: str = config.default_model()
    selected_embedding_provider: str = config.default_embedding_provider()


S = State()
log = logging.getLogger(__name__)
UPLOAD_LOCK = asyncio.Lock()  # one embedding job at a time keeps each chat's index consistent


# ---------- Helpers ----------
def get_thread_index(tid: str):
    """Return (index, chunks) for a chat, loading them from disk on first use."""
    if tid not in S.vectors_per_thread:
        index, chunks = load_thread_index(config.DATA_DIR, tid)
        S.vectors_per_thread[tid] = index
        S.text_per_thread[tid] = chunks
        thread = S.chat_threads.get(tid, {})
        lost = [f['name'] for f in thread.get('processed_files', [])]
        if index is None and lost:
            # The chat lists files but their text is gone: forget them so questions don't silently lack context
            log.warning('Index of chat %s could not be loaded; cleared its files: %s', tid, lost)
            thread['processed_files'] = []
            thread['document_theme'] = ''
            thread.pop('embedding_provider', None)
            save_chat_threads_info(S.chat_threads)
            ui.notify(f"This chat's documents could not be loaded and must be re-uploaded: {', '.join(lost)}",
                      type='warning', timeout=0, close_button=True)
    return S.vectors_per_thread[tid], S.text_per_thread[tid]


def select_thread(tid: str) -> None:
    S.current_thread_id = tid
    get_thread_index(tid)  # load now, so a broken index is reported when the chat is opened
    refresh_all()


def new_thread() -> None:
    """Clear current thread; a fresh one is created on first message."""
    S.current_thread_id = None
    refresh_all()


def refresh_all() -> None:
    chat_title.refresh()
    chats_list.refresh()
    chat_messages.refresh()
    docs_list.refresh()


# ---------- Components ----------
@ui.refreshable
def chat_title() -> None:
    if S.current_thread_id and S.current_thread_id in S.chat_threads:
        name = S.chat_threads[S.current_thread_id].get('name', 'Unnamed Thread')
    else:
        name = 'Hey there!'
    ui.label(name).classes('text-base font-medium text-grey-9')


@ui.refreshable
def chats_list() -> None:
    sorted_threads = sorted(
        S.chat_threads.items(),
        key=lambda kv: kv[1].get('last_updated_at', kv[1].get('created_at', '')),
        reverse=True,
    )
    if not sorted_threads:
        ui.label('No chats yet.').classes('text-grey-6 text-sm')
        return
    for tid, tdata in sorted_threads:
        name = tdata.get('name', 'Unnamed')
        is_selected = tid == S.current_thread_id
        btn_classes = 'w-full justify-start text-left' + (' bg-grey-3' if is_selected else '')
        ui.button(name, on_click=lambda t=tid: select_thread(t)) \
            .props('flat align=left no-caps') \
            .classes(btn_classes)


@ui.refreshable
def chat_messages() -> None:
    if not S.current_thread_id or S.current_thread_id not in S.chat_threads:
        # Empty state — render greeting just above the chat input (it's the last/only thing in the column)
        with ui.column().classes('w-full items-center mt-8'):
            ui.label(get_random_greeting()).classes('italic text-grey-7 text-center max-w-2xl')
        return

    thread = S.chat_threads[S.current_thread_id]
    messages = thread.get('messages', [])

    if not messages:
        with ui.column().classes('w-full items-center mt-8'):
            ui.label(get_random_greeting()).classes('italic text-grey-7 text-center max-w-2xl')
        return

    for msg in messages:
        if msg.get('user'):
            ui.chat_message(msg['user'], name='You', sent=True).classes('w-full')
        if msg.get('answer'):
            ui.chat_message(msg['answer'], name='CORAG').classes('w-full').props('bg-color=blue-1')


@ui.refreshable
def docs_list() -> None:
    if not S.current_thread_id or S.current_thread_id not in S.chat_threads:
        ui.label('Create a chat to upload files.').classes('text-grey-6 text-sm')
        return

    thread = S.chat_threads[S.current_thread_id]
    files = thread.get('processed_files', [])
    if not files:
        ui.label('No files uploaded yet.').classes('text-grey-6 text-sm')
        return

    for f in files:
        with ui.card().classes('w-full q-mb-sm'):
            ui.label(f['name']).classes('font-semibold text-sm')
            ui.label(f['type']).classes('text-xs text-grey-6')


# ---------- File upload ----------
async def handle_upload(e) -> None:
    # NiceGUI 3.x: e.file is a FileUpload with .name, .content_type, and async .read()
    file = e.file
    name = file.name
    content_type = file.content_type or ''

    if not S.current_thread_id:
        tid = str(uuid.uuid4())
        S.current_thread_id = tid
        S.chat_threads[tid] = {
            'name': 'Temporary Thread',
            'messages': [],
            'processed_files': [],
            'document_theme': '',
            'last_updated_at': datetime.now().isoformat(),
            'created_at': datetime.now().isoformat(),
        }
        refresh_all()

    tid = S.current_thread_id  # the user may switch chats while a file is processing
    thread = S.chat_threads[tid]
    if (tid, name) in S.uploads_in_progress or any(f['name'] == name for f in thread.get('processed_files', [])):
        ui.notify(f'{name} already processed', type='info')
        return
    S.uploads_in_progress.add((tid, name))

    notif = ui.notification(f'Processing {name}…', spinner=True, timeout=None)

    try:
        content = await file.read()
        name_lower = name.lower()

        page_starts = None  # PDFs only: where each page begins, so passages can cite their page
        if name_lower.endswith('.pdf'):
            file_like = io.BytesIO(content)
            extracted_text, theme, page_starts = await asyncio.to_thread(process_pdf, file_like)
        elif name_lower.endswith(('.txt', '.md')):
            extracted_text, theme = process_text(content)
        elif name_lower.endswith(('.mp3', '.wav')):
            suffix = '.wav' if name_lower.endswith('.wav') else '.mp3'
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            try:
                extracted_text, theme = await asyncio.to_thread(transcribe_audio, tmp_path)
            finally:
                os.unlink(tmp_path)
        else:
            notif.dismiss()
            ui.notify(f'Unsupported file type: {name}', type='negative')
            return

        if not extracted_text:
            notif.dismiss()
            ui.notify('No text extracted from file.', type='warning')
            return

        # A chat keeps the embedding provider it started with: vectors from different providers can't be mixed
        provider = thread.get('embedding_provider') or S.selected_embedding_provider
        if provider != S.selected_embedding_provider:
            ui.notify(f'This chat already uses {provider} embeddings, so {name} will too.', type='info')

        notif.message = f'Embedding {name}…'
        async with UPLOAD_LOCK:
            index, chunks = get_thread_index(tid)
            index, new_chunks = await asyncio.to_thread(
                process_embeddings, extracted_text, provider, index, name, page_starts,
            )
            chunks = chunks + new_chunks
            S.vectors_per_thread[tid] = index
            S.text_per_thread[tid] = chunks
            save_thread_index(config.DATA_DIR, tid, index, chunks)

        thread.setdefault('processed_files', []).append({'name': name, 'type': content_type})
        thread['document_theme'] = theme
        thread['embedding_provider'] = provider
        save_chat_threads_info(S.chat_threads)
        notif.dismiss()
        ui.notify(f'Processed {name}', type='positive')
        docs_list.refresh()
    except Exception as ex:
        notif.dismiss()
        traceback.print_exc()
        ui.notify(f'Upload error: {ex}', type='negative')
    finally:
        S.uploads_in_progress.discard((tid, name))


# ---------- Chat submission ----------
async def submit_question() -> None:
    question = (input_field.value or '').strip()
    if not question:
        return
    input_field.value = ''

    # Create a thread if none
    if not S.current_thread_id:
        tid = str(uuid.uuid4())
        S.current_thread_id = tid
        S.chat_threads[tid] = {
            'name': 'Temporary Thread',
            'messages': [],
            'processed_files': [],
            'document_theme': '',
            'last_updated_at': datetime.now().isoformat(),
            'created_at': datetime.now().isoformat(),
        }

    thread = S.chat_threads[S.current_thread_id]
    vectors, original_data = get_thread_index(S.current_thread_id)

    # Show user message immediately with a placeholder answer. Keep a handle on this exchange:
    # another question may be appended while this one is being answered. Pending exchanges stay out of chat history.
    exchange = {'user': question, 'answer': '_thinking…_', 'pending': True}
    thread.setdefault('messages', []).append(exchange)
    refresh_all()

    try:
        answer = await asyncio.to_thread(
            process_answer,
            thread_name=S.current_thread_id,
            question=question,
            original_data=original_data,
            vectors=vectors,
            chat_history=get_chat_history(thread, skip_last=True),
            document_theme=thread.get('document_theme', ''),
            inference_model=S.selected_model,
            embedding_provider=thread.get('embedding_provider', S.selected_embedding_provider),
            document_names=[f['name'] for f in thread.get('processed_files', [])],
        )
        exchange['answer'] = answer
        exchange.pop('pending', None)

        if thread.get('name') == 'Temporary Thread':
            try:
                new_name = await asyncio.to_thread(chat_namer, question, answer, S.selected_model)
                thread['name'] = (new_name or 'Untitled').strip().strip('*').strip()
            except Exception:
                traceback.print_exc()

        thread['last_updated_at'] = datetime.now().isoformat()
        save_chat_threads_info(S.chat_threads)
    except Exception as ex:
        exchange['answer'] = f'**Error:** {ex}'
        exchange.pop('pending', None)
        traceback.print_exc()
        try:
            save_chat_threads_info(S.chat_threads)  # otherwise a restart shows "_thinking…_" for this question
        except Exception:
            traceback.print_exc()

    refresh_all()


# ---------- Layout ----------
input_field: ui.input  # forward-declared; assigned in build()


def on_model_change(e) -> None:
    S.selected_model = e.value


def on_embedding_change(e) -> None:
    S.selected_embedding_provider = e.value


# Modern aesthetic — Inter font + custom CSS overriding Quasar's 2010s Material look
MODERN_CSS = '''
:root {
    --bg: #ffffff;
    --bg-subtle: #f7f7f8;
    --bg-muted: #f0f0f1;
    --border: #e5e7eb;
    --text: #1a1a1a;
    --text-muted: #6b7280;
    --accent: #1f2937;
}

html, body, .q-page, .q-layout {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}

/* Header — clean, no Material blue */
.q-header {
    background: rgba(255, 255, 255, 0.85) !important;
    backdrop-filter: saturate(180%) blur(12px);
    -webkit-backdrop-filter: saturate(180%) blur(12px);
    border-bottom: 1px solid var(--border) !important;
    box-shadow: none !important;
    color: var(--text) !important;
}
.q-header .text-white, .q-header * { color: var(--text) !important; }
.q-header .q-separator { background: var(--border) !important; opacity: 1 !important; }

/* Drawers — subtle off-white, no harsh borders */
.q-drawer {
    background: var(--bg-subtle) !important;
    border-color: var(--border) !important;
}
.q-drawer--right { background: var(--bg) !important; }

/* Chat bubbles — minimal, modern */
.q-message {
    margin-bottom: 1.25rem !important;
}
.q-message-name {
    font-size: 0.75rem !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    margin-bottom: 4px !important;
}
.q-message-text {
    border-radius: 18px !important;
    padding: 12px 16px !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
    box-shadow: none !important;
    word-wrap: break-word !important;
}
.q-message-text:before, .q-message-text:after {
    display: none !important;  /* kill the speech-bubble tails */
}
/* User (sent) */
.q-message-sent .q-message-text {
    background: var(--bg-muted) !important;
    color: var(--text) !important;
}
/* AI (received) */
.q-message-received .q-message-text {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
}

/* Footer (chat input) */
.q-footer {
    background: linear-gradient(180deg, transparent 0%, var(--bg) 30%) !important;
    box-shadow: none !important;
    border-top: none !important;
    padding: 0.75rem 1rem !important;
}

/* Pill-shaped input */
.q-field--outlined .q-field__control {
    border-radius: 999px !important;
    padding: 0 8px 0 20px !important;
    background: var(--bg-subtle) !important;
    min-height: 48px !important;
}
.q-field--outlined .q-field__control:before { border: 1px solid var(--border) !important; }
.q-field--outlined.q-field--focused .q-field__control:after { border: 2px solid var(--accent) !important; }

/* Buttons */
.q-btn { text-transform: none !important; font-weight: 500 !important; letter-spacing: 0 !important; }

/* Chat list buttons — minimal */
.q-drawer .q-btn--flat {
    border-radius: 10px !important;
    margin-bottom: 2px !important;
    font-weight: 400 !important;
    color: var(--text) !important;
}
.q-drawer .q-btn--flat:hover { background: var(--bg-muted) !important; }

/* Selects */
.q-field--outlined .q-field__control { background: var(--bg) !important; }
.q-drawer .q-field--outlined .q-field__control { background: var(--bg) !important; }

/* Dividers */
.q-separator { background: var(--border) !important; }

/* Document cards — subtle */
.q-card {
    border-radius: 12px !important;
    box-shadow: none !important;
    border: 1px solid var(--border) !important;
    background: var(--bg) !important;
}

/* Upload widget — tame the Quasar default */
.q-uploader {
    border-radius: 12px !important;
    border: 1px dashed var(--border) !important;
    box-shadow: none !important;
    background: var(--bg) !important;
}
.q-uploader__header {
    background: transparent !important;
    color: var(--text) !important;
    border-bottom: 1px solid var(--border) !important;
}

/* Notifications */
.q-notification { border-radius: 12px !important; }

/* Scrollbar polish */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: #d4d4d8; border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: #a1a1aa; }
::-webkit-scrollbar-track { background: transparent; }
'''


@ui.page('/')
def build() -> None:
    global input_field

    # Inter font + modern overrides
    ui.add_head_html('''
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    ''')
    ui.add_css(MODERN_CSS)

    # Header: logo + app name + dynamic chat title
    with ui.header(bordered=False, elevated=False).classes('items-center gap-3 q-px-md').props('reveal=false height-hint=56'):
        try:
            ui.image('CORAG_ICON.png').classes('w-8 h-8 rounded-full')
        except Exception:
            pass
        ui.label('CORAG').classes('text-lg font-semibold')
        ui.separator().props('vertical').classes('mx-2 opacity-50')
        chat_title()

    # Left drawer: models + chats
    with ui.left_drawer(value=True, bordered=True, fixed=True).classes('p-4').props('width=280'):
        ui.label('Model').classes('text-xs uppercase tracking-wide text-grey-7 q-mb-xs')
        ui.select(
            options=config.models_list,
            value=S.selected_model,
            on_change=on_model_change,
        ).classes('w-full').props('outlined dense')

        ui.label('Embeddings').classes('text-xs uppercase tracking-wide text-grey-7 q-mt-md q-mb-xs')
        ui.select(
            options=config.EMBEDDING_PROVIDERS,
            value=S.selected_embedding_provider,
            on_change=on_embedding_change,
        ).classes('w-full').props('outlined dense')

        ui.separator().classes('q-my-lg')

        with ui.row().classes('items-center justify-between w-full q-mb-sm'):
            ui.label('Chats').classes('text-base font-semibold')
            ui.button(icon='add', on_click=new_thread) \
                .props('round dense flat') \
                .tooltip('New chat')

        with ui.column().classes('w-full gap-0'):
            chats_list()

    # Right drawer: documents
    with ui.right_drawer(value=True, bordered=True, fixed=True).classes('p-4').props('width=300'):
        ui.label('Documents').classes('text-base font-semibold q-mb-md')
        ui.upload(
            on_upload=handle_upload,
            multiple=True,
            auto_upload=True,
            max_file_size=200_000_000,
            label='Drop PDF / TXT / MD / MP3 / WAV',
        ).classes('w-full').props('accept=".pdf,.txt,.md,.mp3,.wav" flat color=grey-7')
        ui.separator().classes('q-my-md')
        with ui.column().classes('w-full gap-2'):
            docs_list()

    # Main content
    with ui.column().classes('w-full max-w-3xl mx-auto q-px-md q-py-lg'):
        chat_messages()

    if not config.has_any_llm_key():
        ui.notify('No model available. Add GOOGLE_API_KEY or ANTHROPIC_API_KEY to your .env file, '
                  'or list local Ollama models in CORAG_OLLAMA_MODELS (see .env.example).',
                  type='warning', timeout=0, close_button=True)

    # Footer: chat input (pill style, send icon inside)
    with ui.footer().props('bordered=false'):
        with ui.row().classes('w-full max-w-3xl mx-auto items-center no-wrap'):
            input_field = ui.input(placeholder='Message CORAG…') \
                .classes('flex-grow') \
                .props('outlined dense rounded borderless')
            input_field.on('keydown.enter', submit_question)
            ui.button(icon='arrow_upward', on_click=submit_question) \
                .props('round dense unelevated color=dark') \
                .classes('q-ml-sm')


ui.run(
    title='CORAG',
    favicon='CORAG_ICON.png',
    show=False,
    reload=False,
    port=8080,
)
