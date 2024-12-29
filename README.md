# Completely OpenSource Retrieval Augmented Generation (CORAG)

This repository implements a Retrieval Augmented Generation (RAG) model enhanced with an Agent-based architecture for contextualizing information retrieval from PDF documents, audio files, and conversational history. The model leverages Large Language Models (currently supporting the Gemini API) integrated with a dynamic toolkit, enabling advanced responses and calculations. Built using Streamlit, this system is designed to provide accurate, context-aware responses.

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

2. Configure API keys in `config.py`:
   - Set your Google Vertex API key from [Google AI Studio](https://aistudio.google.com/)
   - Set your Serper API key from [SerpAPI Dashboard](https://serpapi.com/dashboard) if you intend to use the search tool.

3. Adjust chunk settings in `config.py` as needed.

### Running the Application

To start the server, execute:

```bash
streamlit run server_streamlit.py
```

## Screenshots

![Screenshot](https://github.com/harinaralasetty/Retrieval_Augmented_Generation/blob/main/Screenshot.png)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache-2.0 license.

## Contact

For any questions or feedback, please contact me on LinkedIn: [Hari Naralasetty](https://www.linkedin.com/in/hnaralasetty/)
