# Architecture

CORAG is a single-user local web app. `server.py` builds the NiceGUI interface and calls plain Python modules for ingestion, retrieval, inference and tools. Everything below is taken from the code in this repository.

## Source layout

| Path | Role |
| --- | --- |
| `server.py` | NiceGUI page: model and embedding pickers, chat list, document panel, chat input. Serves on port 8080. |
| `config.py` | Reads `.env`; chunk and HNSW settings, model list, data folder, prompt files. |
| `preprocessing/pdf_processor.py` | Extracts PDF text, detects suspect Indic pages and cleans text while retaining page offsets. |
| `preprocessing/indic_pdf.py` | Optional indic-legacy-pdf adapter; recovers selected pages and reports methods. |
| `preprocessing/text_processor.py` | Decodes `.txt` and `.md` uploads as UTF-8. |
| `preprocessing/audio_processor.py` | Transcribes audio with SpeechRecognition's Google Web Speech recogniser. |
| `preprocessing/prompt_processor.py` | Rephrases questions and names chats with the selected model. |
| `retrieval/embeddings_router.py` | One adapter per embedding provider: Google, Voyage, OpenAI, local sentence-transformers. |
| `retrieval/embeddings_indexing.py` | Chunking, batching, HNSW index build and growth, nearest-neighbour search and re-sorting. |
| `retrieval/index_store.py` | Saves and loads each chat's index (`index.bin`) and chunk texts (`chunks.json`). |
| `inference/inference_manager.py` | `CustomAgentExecutor`: rephrase, retrieve, build the prompt, call tools or the model. |
| `inference/model_router.py` | Sends a prompt to Gemini (LangChain), Claude (Anthropic SDK) or an `ollama/` model (OpenAI client pointed at `OLLAMA_BASE_URL`). |
| `toolkit/` | `calculator` (arithmetic only, parsed with `ast`) and `search` (SerpApi or Tavily). |
| `chat_management/chat_utils.py` | Reads and writes `chat_threads_info.json`; formats chat history. |
| `file_utils.py` | Crash-safe saves: temp file in the same folder, fsync, then rename into place. |
| `prompts/` | The answer, query-rephrasing and chat-naming prompts. |
| `tests/` | pytest suite with fake embeddings and models; drives the real NiceGUI page. |

## Adding a file

1. The upload box accepts `.pdf`, `.txt`, `.md`, `.mp3` and `.wav`, up to 200 MB each. Uploading into an empty view creates a new chat.
2. PDFs are read with PyPDF2. Text and Markdown files are decoded as UTF-8 (`preprocessing/text_processor.py`): a byte-order mark is dropped and invalid bytes are replaced. WAV goes straight to `speech_recognition.AudioFile`; MP3 is first decoded in memory to 16-bit WAV with `soundfile`, which bundles libsndfile, so no ffmpeg is needed. Recordings longer than 60 minutes are refused, MP3 before it is decoded and WAV from its header. Transcription calls Google's Web Speech service over the network, one request per segment of up to 45 seconds, because the free service rejects or cuts off long requests; segments with no recognisable speech are skipped and the texts are joined with spaces.
3. The first 2,000 characters become the chat's "document theme", used later to rephrase questions.
4. Text is split into 1,000-character chunks with 200 characters of overlap (`CHUNK_SETTINGS` in `config.py`).
5. Chunks are embedded in batches of 100 with the chat's embedding provider. A chat keeps the provider it started with, because vectors from different providers cannot be mixed.
6. Vectors go into an `hnswlib` cosine index (`M=30`, `ef_construction=500`), which grows as files are added. Each stored chunk is prefixed with `[Source: <file name>]` so answers can cite it; PDF chunks add the page, `[Source: <file name>, page 3]` or `pages 3-4` when a chunk crosses a page break. The PDF cleanup keeps an offset map (`clean_with_offsets` in `preprocessing/pdf_processor.py`) so page starts are exact in the cleaned text. The page is the page's position in the file, which can differ from the number printed on it (for example after roman-numeral front matter). Chats indexed before this change keep their old labels.
7. The index and chunks are saved under `data/<chat id>/`, and the chat record under `data/chat_threads_info.json`.

## Legacy Indic PDFs

Install with `python -m pip install -r requirements-indic.txt`, then run
`python server.py` and upload a PDF. The optional requirements pin the
indic-legacy-pdf fork and its `glyph` extra. Normal installs and the default
Docker image do not need it.

After PyPDF2 extraction, `likely_broken_indic` checks each page. Readable Indic
Unicode takes precedence over font names. A high share of Latin-1 or private-use
characters, known legacy font names, or Telugu Identity-H/V fonts without
ToUnicode can trigger recovery. PyPDF2's text visitor reports fonts actually used by extracted text, including
Form text, so unused legacy font resources do not trigger recovery. Ordinary English and proper Unicode
Telugu pages bypass the converter.

The adapter imports the library lazily and writes the uploaded bytes to a private
temporary PDF because embedded-font recovery requires a file path. It uses
`PDFParser`, `EncodingDetector.detect_from_document` and
`UnicodeConverter.convert_with_recovery`. This supports Telugu glyph-ID recovery
and built-in legacy mappings such as Kruti Dev, Shree-Dev and DVB-TT for
Hindi/Marathi. It makes no OCR, translation or network requests. The temporary
file is deleted even on failure.

Only suspect pages with more Indic Unicode and at most 5% replacement characters
replace their PyPDF2 text. Other pages retain their original extraction. Cleanup
and page-start offset mapping run after replacement, so blank pages and chunk
citations keep their original page numbers. Import/conversion errors and
unhelpful output fall back to PyPDF2; methods and converter warnings are logged.
Glyph ambiguity and approximate reading order can still affect recovered text.

`process_pdf` keeps its three-value return contract. The upload handler passes a
per-upload `recovery_info` dictionary, then stores its label, methods and 1-based
page numbers in the file record under `processed_files[].recovery`. Documents
renders the saved label after uploads and restarts. Older file records have no
recovery field and continue to render normally.

`python scripts/demo_indic_pdf.py` exercises the original synthetic fixture,
recovery, chunking, HNSW search and page citations offline. Tests also cover the
missing-library path, conversion failures and unmodified readable PDFs. The
fixture font and PDF content are generated locally; no external PDFs are used.

## Asking a question

1. The selected model rephrases the question using the document theme (`prompts/query_rephrasal_prompt.txt`).
2. The rephrased question is embedded with the chat's provider.
3. The 20 nearest chunks (or fewer) are fetched from the HNSW index and re-sorted by cosine similarity.
4. `prompts/base_prompt.txt` is filled with the passages, the uploaded file names, the chat's earlier exchanges and the agent's last five messages.
5. Gemini models get the `calculator` and `search` tools bound through LangChain. If the model calls a tool, its output is added to the prompt and the model is asked again. If the tool chain fails, the plain prompt is sent to Gemini instead.
6. Claude and Ollama models get the plain prompt with no tools.
7. After the first answer in a new chat, the model names the chat (`prompts/chat_naming.txt`).

Uploads and questions run on worker threads, so a question can search a chat's index while an upload grows it. Each index has a lock, held for one add (including any resize) or for one question's search; a question only uses the chunks that existed when it was asked, and exchanges still waiting for an answer are left out of the chat history.

A failed model call raises an error whose message has API keys removed. If rephrasing fails, the question is searched as asked and a warning is logged; if naming fails, the chat keeps its temporary name and naming is tried again after the next answer; if answering fails, the chat shows and saves `**Error:**` with the cause.

## Saved data

Saves write a temp file in the same folder, fsync it and rename it over the old file, so a crash leaves the old or the new version. A chat's chunk texts are renamed after its index, and loading checks that the index holds one vector per chunk. If a chat still lists files but its index is missing, unreadable or mismatched, opening the chat logs a warning, clears its file list, document theme and embedding provider on disk, and shows a warning (until closed) naming the files to upload again. The chat and its messages stay usable, and the next upload builds a fresh index that replaces the broken files.

## Limits

- With Gemini or Claude, the question and retrieved passages are sent to Google or Anthropic. Ollama models run wherever `OLLAMA_BASE_URL` points, by default this computer.
- No OCR: scanned PDFs without a text layer produce no text.
- Retrieval sends up to 20 chunks to the model for every question, regardless of how relevant they are.
- App state is shared by every open browser tab; it is not built for several users.
