import json

import numpy as np
import pytest

import config
from retrieval import embeddings_indexing as ei
from retrieval.index_store import load_thread_index, save_thread_index


def vec(text):
    v = np.zeros(26, dtype=np.float32) + 1e-3
    for ch in text.lower():
        if "a" <= ch <= "z":
            v[ord(ch) - 97] += 1
    return (v / np.linalg.norm(v)).tolist()


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [vec(t) for t in texts]


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    monkeypatch.setattr(ei, "get_embeddings_model", lambda provider: FakeEmbeddings())


@pytest.fixture
def small_chunks(monkeypatch):
    monkeypatch.setitem(config.CHUNK_SETTINGS, "CHUNK_SIZE", 40)
    monkeypatch.setitem(config.CHUNK_SETTINGS, "CHUNK_OVERLAP", 10)


def test_short_text_is_one_chunk():
    assert ei._chunk_overlapping("hello", 1000, 200) == ["hello"]
    assert ei._chunk_overlapping("", 1000, 200) == []


def test_chunks_cover_the_whole_text_with_overlap():
    text = "".join(chr(97 + i % 26) for i in range(2500))
    chunks = ei._chunk_overlapping(text, 1000, 200)
    assert chunks[0] == text[:1000]
    assert chunks[1].startswith(chunks[0][-200:])
    assert chunks[-1].endswith(text[-1])
    assert all(len(c) <= 1000 for c in chunks)


def test_second_document_extends_the_same_index(small_chunks):
    index, chunks_a = ei.generate_embeddings("aaaa bbbb " * 20, source="a.txt")
    index, chunks_b = ei.generate_embeddings("zzzz qqqq " * 20, vectors=index, source="b.txt")
    texts = chunks_a + chunks_b
    assert index.get_current_count() == len(texts)
    assert all(c.startswith("[Source: a.txt]") for c in chunks_a)
    assert all(c.startswith("[Source: b.txt]") for c in chunks_b)
    context = ei.fetch_relevant_data(ei.generate_query_embedding("zzzz qqqq"), index, texts)
    assert context.split("\n\n")[0].startswith("[Source: b.txt]")


def test_index_grows_past_initial_capacity(monkeypatch):
    monkeypatch.setitem(config.CHUNK_SETTINGS, "CHUNK_SIZE", 10)
    monkeypatch.setitem(config.CHUNK_SETTINGS, "CHUNK_OVERLAP", 0)
    index, first = ei.generate_embeddings("abcdefghij" * 50)
    index, second = ei.generate_embeddings("klmnopqrst" * 1500, vectors=index)
    assert index.get_current_count() == len(first) + len(second) > 1024


def test_index_roundtrip_on_disk(tmp_path, small_chunks):
    index, chunks = ei.generate_embeddings("hello world " * 30, source="h.txt")
    save_thread_index(str(tmp_path), "t1", index, chunks)
    loaded, loaded_chunks = load_thread_index(str(tmp_path), "t1")
    assert loaded_chunks == chunks
    q = ei.generate_query_embedding("hello world")
    assert ei.fetch_relevant_data(q, loaded, loaded_chunks) == ei.fetch_relevant_data(q, index, chunks)


def test_missing_or_corrupt_index_loads_as_empty(tmp_path, small_chunks):
    assert load_thread_index(str(tmp_path), "nope") == (None, [])
    index, chunks = ei.generate_embeddings("hello world " * 30)
    save_thread_index(str(tmp_path), "t2", index, chunks)
    meta = tmp_path / "t2" / "chunks.json"
    data = json.loads(meta.read_text())
    data["texts"] = data["texts"][:-1]
    meta.write_text(json.dumps(data))
    assert load_thread_index(str(tmp_path), "t2") == (None, [])


def test_chunks_added_while_a_question_is_answered_are_ignored(small_chunks):
    # The question holds the chunk list it started with; an upload meanwhile grows the shared index in place
    index, chunks = ei.generate_embeddings("aaaa bbbb " * 20, source="a.txt")
    ei.generate_embeddings("zzzz qqqq " * 20, vectors=index, source="b.txt")
    context = ei.fetch_relevant_data(ei.generate_query_embedding("zzzz qqqq"), index, chunks)
    assert context and "[Source: b.txt]" not in context
