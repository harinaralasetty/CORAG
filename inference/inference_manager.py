from langchain.memory import ConversationBufferMemory
from preprocessing.prompt_processor import rephrase_prompt
from inference.gemini_interaction import generate_gemini_response, get_model_instance
from toolkit.tool_manager import call_tools, tools
from config import BASE_PROMPT
from retrieval.embeddings_indexing import (
    generate_query_embedding,
    fetch_relevant_data,
    generate_embeddings,
)
import traceback

# Dictionary to hold one agent per thread
agents = {}

class CustomAgentExecutor:
    """
    Custom agent that can handle queries, optionally use tools,
    and track conversation memory.
    """
    def __init__(
        self,
        tool_manager_func,
        llm,
        llm_with_tools,
        memory,
        vectors=None,
        original_data=None
    ):
        self.tool_manager_func = tool_manager_func
        self.llm = llm
        self.llm_with_tools = llm_with_tools
        self.memory = memory

        # Store any available vectors and original_data
        self.vectors = vectors or []
        self.original_data = original_data or []

    def run(self, user_input, document_theme=None, chat_history=None, additional_vectors=None, inference_model = None):
        """
        Process the user input and generate a response using the model (Gemini).
        Tools may be invoked if needed.
        """
        # Step 1: Store user input in memory
        self.memory.chat_memory.add_user_message(user_input)

        # Step 2: Rephrase the input
        rephrased_input = rephrase_prompt(user_input, document_theme, inference_model)
        rephrased_input = user_input

        # Step 3: Fetch relevant data for context
        query_vector = generate_query_embedding(rephrased_input)

        if not self.vectors: relevant_context = "";
        else: relevant_context = fetch_relevant_data(query_vector, self.vectors, self.original_data);

        relevant_chat_history = chat_history or ""

        # Step 4: Construct the dynamic prompt
        prompt = BASE_PROMPT.format(
            context=relevant_context,
            question=rephrased_input,
            relevant_chat_history=relevant_chat_history,
            last_n=5,
            last_n_messages=self.memory.chat_memory.messages[-5:],
            tool_call=""
        )

        # Step 5: Tool + LLM interaction
        try:
            chain = self.llm_with_tools | (lambda msg: self.tool_manager_func(msg, self.llm))
            response = chain.invoke(prompt)

            # Step 6: Store the response
            self.memory.chat_memory.add_ai_message(response)
            return response
        except Exception as e:
            # Fallback to plain Gemini
            print(f"[DEBUG] Tool-based execution failed, falling back to Gemini. Error: {e}")
            print(traceback.format_exc())
            response = generate_gemini_response(prompt, inference_model)
            self.memory.chat_memory.add_ai_message(response)
            return response


def initialize_agent(thread_name, original_data=None, vectors=None, inference_model = None):
    """
    Create or update a CustomAgentExecutor for a given thread.
    We assume embeddings were generated outside and passed in as 'vectors'.
    """
    key_name = thread_name + "_" + inference_model
    if key_name not in agents:
        # Instantiate LLM with tools integrated
        print(f"IM", inference_model)
        llm = get_model_instance(inference_model)
        llm_with_tools = llm.bind_tools(tools)

        # Create fresh conversation memory
        memory = ConversationBufferMemory(memory_key="chat_history")

        agent = CustomAgentExecutor(
            tool_manager_func=call_tools,
            llm=llm,
            llm_with_tools=llm_with_tools,
            memory=memory,
            vectors=vectors,
            original_data=original_data
        )
        agents[key_name] = agent
    else:
        # Update existing agent if needed
        agent = agents[key_name]
        if original_data:
            # Extend the original_data
            agent.original_data.extend(original_data)
        if vectors:
            agent.vectors = vectors
    return agents[key_name]


def process_answer(
    BASE_PROMPT,
    thread_name,
    question,
    original_data=None,
    vectors=None,
    chat_history=None,
    additional_vectors=None,
    document_theme=None,
    inference_model = None
):
    """
    Main entry point for generating an answer to user's question:
    1) Initialize/update the agent with new data/vectors.
    2) Pass the question to the agent's run() method.
    """
    # 1. Initialize or update the agent for this thread
    agent = initialize_agent(
        thread_name,
        original_data=original_data,
        vectors=vectors, 
        inference_model = inference_model
    )

    # 2. Run the agent
    print(f"\n[DEBUG] Processing Question: {question}")
    response = agent.run(
        user_input=question,
        document_theme=document_theme,
        chat_history=chat_history,
        additional_vectors=additional_vectors,
        inference_model = inference_model
    )

    print(f"\n[DEBUG] Agent Response: {response}\n")
    return response.content if hasattr(response, 'content') else response


def process_embeddings(text_list):
    """
    Sole function to generate embeddings from raw text segments.
    """
    if not text_list:
        return []
    return generate_embeddings(text_list)
