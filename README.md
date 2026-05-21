# Completely OpenSource Retrieval Augmented Generation (CORAG)

This repository implements a Retrieval Augmented Generation (RAG) model enhanced with an Agent-based architecture for contextualizing information retrieval from PDF documents, audio files, and conversational history. The model leverages Large Language Models (Google Gemini and Anthropic Claude) integrated with a dynamic toolkit, enabling advanced responses and calculations. Embeddings can be generated via Google, Voyage AI, OpenAI, or a local sentence-transformers model. Built with NiceGUI for a modern chat interface.

## Overview

The RAG model processes both PDF documents and audio files by first converting their content into numerical representations (embeddings). It uses Hierarchical Navigable Small World (HNSW) indexing with cosine similarity to retrieve the most relevant sections of the input in response to a user's query. Additionally, the system employs a Custom Agent Executor to dynamically decide whether to retrieve information, call external tools, or directly generate responses using the LLM. This ensures more intelligent and contextualized interactions.

## Key Features
- Agent-Driven Architecture: The agent acts as the central decision-maker, combining LLM capabilities with external tools and memory for enhanced contextuality.
- Dynamic Prompting: The system constructs adaptive prompts using conversational memory, embeddings, and retrieved data for improved response relevance.
- Toolkit Integration: Includes external tools such as search and calculator functionalities, invoked dynamically by the agent.
- PDF & Audio Processing: Converts document and audio content into embeddings for retrieval and contextualization.
- Memory Management: Incorporates conversational memory to ensure responses are consistent with prior interactions.
- HNSW Indexing & Reranking: Utilizes efficient retrieval methods and reranking mechanisms for higher accuracy.

## Flow

![Flowchart](https://github.com/harinaralasetty/Retrieval_Augmented_Generation/blob/main/Flowchart.png)

## Getting Started

### Prerequisites

- Python environment with necessary dependencies.

### Installation

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project root with your API keys:
   ```
   GOOGLE_API_KEY=your_google_key       # from https://aistudio.google.com/
   SERPER_API_KEY=your_serper_key       # from https://serpapi.com/dashboard (search tool)
   ANTHROPIC_API_KEY=your_anthropic_key # optional, for Claude models — https://console.anthropic.com/
   VOYAGE_API_KEY=your_voyage_key       # optional, for Voyage embeddings — https://dash.voyageai.com/
   OPENAI_API_KEY=your_openai_key       # optional, for OpenAI embeddings — https://platform.openai.com/
   ```
   Only `GOOGLE_API_KEY` and `SERPER_API_KEY` are required. The rest are optional and only need to be set if you select that provider in the UI. For local embeddings (`sentence-transformers`), no key is needed.

3. Adjust chunk settings in `config.py` as needed.

### Running the Application

To start the server, execute:

```bash
python server.py
```

Then open [http://localhost:8080](http://localhost:8080) in your browser.

## Screenshots

![Screenshot](https://github.com/harinaralasetty/Retrieval_Augmented_Generation/blob/main/Screenshot.png)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache-2.0 license.

## Contact

For any questions or feedback, please contact me on LinkedIn: [Hari Naralasetty](https://www.linkedin.com/in/harinaralasetty/)
