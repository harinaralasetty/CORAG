import numpy as np
import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import config
from inference import gemini_interaction as gi
from inference import inference_manager as im
from preprocessing import prompt_processor as pp
from retrieval import embeddings_indexing as ei


def vec(text):
    v = np.zeros(26, dtype=np.float32) + 1e-3
    for ch in text.lower():
        if "a" <= ch <= "z":
            v[ord(ch) - 97] += 1
    return (v / np.linalg.norm(v)).tolist()


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [vec(t) for t in texts]


class RecordingLLM:
    """Stands in for the final LLM call; records the prompt it receives."""

    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return AIMessage(content="final answer")


@pytest.fixture(autouse=True)
def patches(monkeypatch):
    monkeypatch.setattr(ei, "get_embeddings_model", lambda provider: FakeEmbeddings())
    monkeypatch.setattr(im, "rephrase_prompt", lambda q, theme, model: q)
    monkeypatch.setattr(im, "generate_query_embedding", ei.generate_query_embedding)
    monkeypatch.setitem(config.CHUNK_SETTINGS, "CHUNK_SIZE", 40)
    monkeypatch.setitem(config.CHUNK_SETTINGS, "CHUNK_OVERLAP", 10)
    im.agents.clear()


def make_agent(tool_calls, final_llm):
    llm_with_tools = RunnableLambda(lambda prompt: AIMessage(content="", tool_calls=tool_calls))
    return im.CustomAgentExecutor(
        tool_manager_func=im.call_tools,
        llm=final_llm,
        llm_with_tools=llm_with_tools,
        memory=im.ConversationBufferMemory(memory_key="chat_history"),
    )


def test_tool_path_keeps_question_and_document_context():
    index, chunks = ei.generate_embeddings("the launch code is zebra " * 5, source="notes.txt")
    final = RecordingLLM()
    agent = make_agent([{"name": "calculator", "args": {"expression": "6*7"}, "id": "1"}], final)
    agent.vectors, agent.original_data = index, chunks
    answer = agent.run("What is 6*7 and the launch code?", inference_model="gemini-2.5-flash")
    assert answer == "final answer"
    prompt = final.prompts[0]
    assert "What is 6*7 and the launch code?" in prompt
    assert "[Source: notes.txt]" in prompt and "zebra" in prompt
    assert "calculator(" in prompt and "42" in prompt


def test_no_documents_still_answers():
    final = RecordingLLM()
    agent = make_agent([], final)
    result = agent.run("hello", inference_model="gemini-2.5-flash")
    assert result == ""  # no tool calls: the model's own (empty) content is returned


def test_reusing_an_agent_replaces_instead_of_extending_chunks():
    index, chunks = ei.generate_embeddings("alpha beta gamma " * 5)
    for _ in range(3):
        agent = im.initialize_agent("t", original_data=chunks, vectors=index, inference_model="claude-haiku-4-5-20251001")
    assert agent.original_data == chunks and len(agent.original_data) == len(chunks)
    agent = im.initialize_agent("t", original_data=None, vectors=None, inference_model="claude-haiku-4-5-20251001")
    assert agent.original_data == [] and agent.vectors is None


class BrokenModel:
    def invoke(self, prompt):
        raise RuntimeError("quota exceeded")


def test_a_failed_rephrase_searches_with_the_question_as_asked(monkeypatch, caplog):
    monkeypatch.setattr(im, "rephrase_prompt", pp.rephrase_prompt)
    monkeypatch.setattr(gi, "get_model_instance", lambda model: BrokenModel())
    queries = []

    def record_query(text, provider):
        queries.append(text)
        return ei.generate_query_embedding(text, provider)

    monkeypatch.setattr(im, "generate_query_embedding", record_query)
    agent = make_agent([], RecordingLLM())
    agent.run("What is the launch code?", inference_model="gemini-2.5-flash")
    assert queries == ["What is the launch code?"]
    assert "quota exceeded" in caplog.text


def test_ollama_models_get_the_plain_prompt_without_tools(monkeypatch):
    sent = []
    monkeypatch.setattr(im, "generate_response", lambda prompt, model: sent.append((model, prompt)) or "local answer")
    index, chunks = ei.generate_embeddings("the launch code is zebra " * 5, source="notes.txt")
    agent = im.initialize_agent("t", original_data=chunks, vectors=index, inference_model="ollama/llama3.2")
    assert agent.llm is None and agent.llm_with_tools is None
    assert agent.run("What is the launch code?", inference_model="ollama/llama3.2") == "local answer"
    model, prompt = sent[0]
    assert model == "ollama/llama3.2"
    assert "[Source: notes.txt]" in prompt and "zebra" in prompt
