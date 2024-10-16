**Retreival Augmented Generation**

This repository implements a Response Augmentation Generation (RAG) model for contextualizing information retrieval from PDFs. Built using Gradio, the model takes a PDF document as input and leverages the Gemini API to generate a response.

The model first processes the PDF through an embedding layer, converting textual information into a numerical representation. HNSW indexing with cosine similarity is then employed to identify sections within the PDF that hold the most relevant context to the user's query. This contextualization process even incorporates chat history, ensuring the response considers the entire conversation thread. Reranking is added for further enhancement in retrieval. 

Flow: 
![alt text](https://github.com/harinaralasetty/Retrieval_Augmented_Generation/blob/main/Flowchart.png)

To run: 
1. Install requirements on your python env.
2. In config.py set Google Vertex API key and Serper API key from https://serpapi.com/dashboard (if intended to use search tool).
3. Further configure the chunk settings as necessary. 
4. Run server.py

![alt text](https://github.com/harinaralasetty/Retrieval_Augmented_Generation/blob/main/Screenshot.png)
