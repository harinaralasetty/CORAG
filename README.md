This repository implements a Response Augmentation Generation (RAG) model for contextualizing information retrieval from PDFs. Built using Gradio, the model takes a PDF document as input and leverages the Gemini API to generate a response.

The model first processes the PDF through an embedding layer, converting textual information into a numerical representation. Cosine similarity is then employed to identify sections within the PDF that hold the most relevant context to the user's query. This contextualization process even incorporates chat history, ensuring the response considers the entire conversation thread.

Flow: 
![flow chart](https://github.com/harinaralasetty/Retrieval_Augmented_Generation/blob/main/Lightweight%20RAG%20(1).png)

To run: 
1. Install requirements on your python env.
2. Get Google Vertex API key from https://aistudio.google.com/app/apikey and insert in config.json
3. Run server.py

![alt text](https://github.com/harinaralasetty/Retrieval_Augmented_Generation/blob/main/RAG%20Local%20Screenshot.png)
