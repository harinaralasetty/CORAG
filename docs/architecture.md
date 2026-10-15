# Architecture

CORAG is a single-user local web app. `server.py` builds the NiceGUI interface and calls plain Python modules for ingestion, retrieval, inference and tools. Everything below is taken from the code in this repository.

## Source layout

| Path | Role |
| --- | --- |
| `server.py` | NiceGUI page: model and embedding pickers, chat list, document panel, chat input. Serves on port 8080. |
| `config.py` | Reads `.env`; chunk and HNSW settings, model list, data folder, prompt files. |
| `preprocessing/pdf_processor.py` | Extracts PDF text with PyPDF2 and cleans ligatures and broken lines. |
| `preprocessing/audio_processor.py` | Transcribes audio with SpeechRecognition's Google Web Speech recogniser. |
| `preprocessing/prompt_processor.py` | Rephrases questions and names chats with the selected model. |
| `retrieval/embeddings_router.py` | One adapter per embedding provider: Google, Voyage, OpenAI, local sentence-transformers. |
| `retrieval/embeddings_indexing.py` | Chunking, batching, HNSW index build and growth, nearest-neighbour search and re-sorting. |
| `retrieval/index_store.py` | Saves and loads each chat's index (`index.bin`) and chunk texts (`chunks.json`). |
| `inference/inference_manager.py` | `CustomAgentExecutor`: rephrase, retrieve, build the prompt, call tools or the model. |
| `inference/model_router.py` | Sends a prompt to Gemini (LangChain) or Claude (Anthropic SDK). |
| `toolkit/` | `calculator` (arithmetic only, parsed with `ast`) and `search` (SerpApi or Tavily). |
| `chat_management/chat_utils.py` | Reads and writes `chat_threads_info.json`; formats chat history. |
| `file_utils.py` | Crash-safe saves: temp file in the same folder, fsync, then rename into place. |
| `prompts/` | The answer, query-rephrasing and chat-naming prompts. |
| `tests/` | pytest suite with fake embeddings and models; drives the real NiceGUI page. |

## Adding a file

1. The upload box accepts `.pdf`, `.mp3` and `.wav`, up to 200 MB each. Uploading into an empty view creates a new chat.
2. PDFs are read with PyPDF2. WAV goes straight to `speech_recognition.AudioFile`; MP3 is first decoded in memory to 16-bit WAV with `soundfile`, which bundles libsndfile, so no ffmpeg is needed. Recordings longer than 60 minutes are refused, MP3 before it is decoded and WAV from its header. Transcription calls Google's Web Speech service over the network, one request per segment of up to 45 seconds, because the free service rejects or cuts off long requests; segments with no recognisable speech are skipped and the texts are joined with spaces.
3. The first 2,000 characters become the chat's "document theme", used later to rephrase questions.
4. Text is split into 1,000-character chunks with 200 characters of overlap (`CHUNK_SETTINGS` in `config.py`).
5. Chunks are embedded in batches of 100 with the chat's embedding provider. A chat keeps the provider it started with, because vectors from different providers cannot be mixed.
6. Vectors go into an `hnswlib` cosine index (`M=30`, `ef_construction=500`), which grows as files are added. Each stored chunk is prefixed with `[Source: <file name>]` so answers can cite it.
7. The index and chunks are saved under `data/<chat id>/`, and the chat record under `data/chat_threads_info.json`.

## Asking a question

1. The selected model rephrases the question using the document theme (`prompts/query_rephrasal_prompt.txt`).
2. The rephrased question is embedded with the chat's provider.
3. The 20 nearest chunks (or fewer) are fetched from the HNSW index and re-sorted by cosine similarity.
4. `prompts/base_prompt.txt` is filled with the passages, the uploaded file names, the chat's earlier exchanges and the agent's last five messages.
5. Gemini models get the `calculator` and `search` tools bound through LangChain. If the model calls a tool, its output is added to the prompt and the model is asked again. If the tool chain fails, the plain prompt is sent to Gemini instead.
6. Claude models get the plain prompt with no tools.
7. After the first answer in a new chat, the model names the chat (`prompts/chat_naming.txt`).

## Saved data

Saves write a temp file in the same folder, fsync it and rename it over the old file, so a crash leaves the old or the new version. A chat's chunk texts are renamed after its index, and loading checks that the index holds one vector per chunk. If a chat still lists files but its index is missing, unreadable or mismatched, opening the chat logs a warning, clears its file list, document theme and embedding provider on disk, and shows a warning (until closed) naming the files to upload again. The chat and its messages stay usable, and the next upload builds a fresh index that replaces the broken files.

## Limits

- Answers come from cloud models; the question and retrieved passages are sent to Google or Anthropic.
- No OCR: scanned PDFs without a text layer produce no text.
- Retrieval sends up to 20 chunks to the model for every question, regardless of how relevant they are.
- App state is shared by every open browser tab; it is not built for several users.
