import logging
import traceback

from langchain.memory import ConversationBufferMemory

from config import BASE_PROMPT
from inference.gemini_interaction import generate_gemini_response, get_model_instance
from inference.llm_error import LLMError
from inference.model_router import generate_response, is_claude
from preprocessing.prompt_processor import rephrase_prompt
from retrieval.embeddings_indexing import (
    fetch_relevant_data,
    generate_embeddings,
    generate_query_embedding,
)
from toolkit.tool_manager import call_tools, tools

# One agent per (thread, model) pair
agents = {}
log = logging.getLogger(__name__)


class CustomAgentExecutor:
    """Custom agent that handles queries, optionally uses tools, and tracks conversation memory."""

    def __init__(self, tool_manager_func, llm, llm_with_tools, memory, vectors=None, original_data=None):
        self.tool_manager_func = tool_manager_func
        self.llm = llm
        self.llm_with_tools = llm_with_tools
        self.memory = memory
        self.vectors = vectors
        self.original_data = original_data or []
        # Provider used to build the index — query embeddings must match
        self.embedding_provider = "google"

    def run(self, user_input, document_theme=None, chat_history=None, inference_model=None, document_names=None):
        self.memory.chat_memory.add_user_message(user_input)

        try:
            rephrased_input = rephrase_prompt(user_input, document_theme, inference_model)
        except LLMError as e:
            log.warning("Could not rephrase the question, searching with it as asked: %s", e)
            rephrased_input = user_input

        query_vector = generate_query_embedding(rephrased_input, self.embedding_provider)

        relevant_context = (
            fetch_relevant_data(query_vector, self.vectors, self.original_data)
            if self.vectors is not None and self.original_data else ""
        )

        def build_prompt(tool_output=""):
            return BASE_PROMPT.format(
                context=relevant_context,
                question=rephrased_input,
                relevant_chat_history=chat_history or "",
                last_n=5,
                last_n_messages=self.memory.chat_memory.messages[-5:],
                tool_call=tool_output,
                document_names=", ".join(document_names) if document_names else "none",
            )

        prompt = build_prompt()

        # Claude path: skip langchain tool chain (no langchain-anthropic to avoid
        # version conflicts with langchain-google-genai). Plain message → response.
        if is_claude(inference_model):
            response = generate_response(prompt, inference_model)
            self.memory.chat_memory.add_ai_message(response)
            return response

        try:
            chain = self.llm_with_tools | (lambda msg: self.tool_manager_func(msg, self.llm, build_prompt))
            response = chain.invoke(prompt)
            self.memory.chat_memory.add_ai_message(response)
            return response
        except Exception as e:
            print(f"[DEBUG] Tool-based execution failed, falling back to Gemini. Error: {e}")
            print(traceback.format_exc())
            response = generate_gemini_response(prompt, inference_model)
            self.memory.chat_memory.add_ai_message(response)
            return response


def initialize_agent(thread_name, original_data=None, vectors=None, inference_model=None, embedding_provider="google"):
    """Create or update a CustomAgentExecutor for a given thread."""
    key_name = f"{thread_name}_{inference_model}"

    if key_name not in agents:
        if is_claude(inference_model):
            # Claude path bypasses the langchain tool chain entirely
            llm = None
            llm_with_tools = None
        else:
            llm = get_model_instance(inference_model)
            llm_with_tools = llm.bind_tools(tools)

        agents[key_name] = CustomAgentExecutor(
            tool_manager_func=call_tools,
            llm=llm,
            llm_with_tools=llm_with_tools,
            memory=ConversationBufferMemory(memory_key="chat_history"),
            vectors=vectors,
            original_data=original_data,
        )
    else:
        # The caller owns the full chunk list and index for the thread, so replace (never extend)
        agent = agents[key_name]
        agent.original_data = original_data or []
        agent.vectors = vectors

    agents[key_name].embedding_provider = embedding_provider
    return agents[key_name]


def process_answer(
    thread_name,
    question,
    original_data=None,
    vectors=None,
    chat_history=None,
    document_theme=None,
    inference_model=None,
    embedding_provider="google",
    document_names=None,
):
    """Main entry point for generating an answer."""
    agent = initialize_agent(
        thread_name,
        original_data=original_data,
        vectors=vectors,
        inference_model=inference_model,
        embedding_provider=embedding_provider,
    )

    response = agent.run(
        user_input=question,
        document_theme=document_theme,
        chat_history=chat_history,
        inference_model=inference_model,
        document_names=document_names,
    )

    return response.content if hasattr(response, 'content') else response


def process_embeddings(text, embedding_provider="google", vectors=None, source=None):
    """Chunk and embed text, adding it to an existing index when one is given.

    Returns (index, chunks) where chunks are only the newly created ones.
    """
    if not text:
        return vectors, []
    return generate_embeddings(text, vectors=vectors, embedding_provider=embedding_provider, source=source)
