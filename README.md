# Completely OpenSource Retrieval Augmented Generation (CORAG)

This repository implements a Retrieval Augmented Generation (RAG) model enhanced with an Agent-based architecture for contextualizing information retrieval from PDF documents, audio files, and conversational history. The model leverages Large Language Models (Google Gemini and Anthropic Claude) integrated with a dynamic toolkit, enabling advanced responses and calculations. Embeddings can be generated via Google, Voyage AI, OpenAI, or a local sentence-transformers model. Built with NiceGUI for a modern chat interface.

## Overview

The RAG model processes both PDF documents and audio files by first converting their content into numerical representations (embeddings). It uses Hierarchical Navigable Small World (HNSW) indexing with cosine similarity to retrieve the most relevant sections of the input in response to a user's query. Additionally, the system employs a Custom Agent Executor to dynamically decide whether to retrieve information, call external tools, or directly generate responses using the LLM. This ensures more intelligent and contextualized interactions.

## Key Features
- Agent-Driven Architecture: The agent acts as the central decision-maker, combining LLM capabilities with external tools and memory for enhanced contextuality.
- Dynamic Prompting: The system constructs adaptive prompts using conversational memory, embeddings, and retrieved data for improved response relevance.
- Toolkit Integration: Includes external tools such as search and calculator functionalities, invoked dynamically by the agent.
- PDF & Audio Processing: Converts document and audio content into embeddings for retrieval and contextualization.
- Source Labels: Retrieved passages carry the name of the file they came from, so answers can cite the right document.
- Memory Management: Incorporates conversational memory to ensure responses are consistent with prior interactions.
- HNSW Indexing & Reranking: Utilizes efficient retrieval methods and reranking mechanisms for higher accuracy.

## Flow

![Flowchart](Flowchart.png)

## Getting Started

### Prerequisites

- Python 3.13 (what the project is developed and tested on)
- An API key for Google Gemini or Anthropic Claude (see `.env.example`)

### Installation

1. Create a virtual environment and install the dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   `requirements.txt` lists the direct dependencies; `requirements-lock.txt` has the exact pinned set.

2. Copy `.env.example` to `.env` and add your keys:
   ```bash
   cp .env.example .env
   ```
   You need at least one of `GOOGLE_API_KEY` or `ANTHROPIC_API_KEY`. Embeddings use Google, Voyage or OpenAI when their key is set, otherwise a local `sentence-transformers` model that needs no key. The web search tool is optional: set `SEARCH_PROVIDER` to `serpapi` (needs `SERPER_API_KEY`) or `tavily` (needs `TAVILY_API_KEY`).

3. Adjust chunk settings in `config.py` as needed (default: 1000 characters, 200 overlap).

### Running the Application

To start the server, execute:

```bash
python server.py
```

Then open [http://localhost:8080](http://localhost:8080) in your browser.

Chats and their document indexes are saved in `data/` (set `CORAG_DATA_DIR` to change it), so they are still there after a restart. Delete that folder to start fresh.

### Running the tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests use fake embeddings and models, so they need no API keys.

## Screenshots

![Screenshot](Screenshot.png)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache-2.0 license.

## Contact

For any questions or feedback, please contact me on LinkedIn: [Hari Naralasetty](https://www.linkedin.com/in/harinaralasetty/)
