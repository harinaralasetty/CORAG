import json
import threading

import hnswlib
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


def test_new_chunks_do_not_crowd_out_the_questions_own_chunks(small_chunks):
    # More than 20 newer chunks sit closer to the query than the question's own few chunks
    index, chunks = ei.generate_embeddings("cccc dddd " * 12, source="a.txt")
    ei.generate_embeddings("aaaa bbbb " * 100, vectors=index, source="b.txt")
    assert index.get_current_count() > len(chunks) + 20
    context = ei.fetch_relevant_data(ei.generate_query_embedding("aaaa bbbb"), index, chunks)
    assert "[Source: a.txt]" in context and "[Source: b.txt]" not in context


def test_a_crash_while_saving_keeps_the_previous_index(tmp_path, small_chunks, monkeypatch):
    index, chunks = ei.generate_embeddings("hello world " * 30, source="a.txt")
    save_thread_index(str(tmp_path), "t3", index, chunks)
    index, more = ei.generate_embeddings("zzzz qqqq " * 30, vectors=index, source="b.txt")

    def crash_midway(obj, fp, **kwargs):
        fp.write('{"dim": 26, "te')
        raise OSError("disk full")

    with monkeypatch.context() as m, pytest.raises(OSError):
        m.setattr(json, "dump", crash_midway)
        save_thread_index(str(tmp_path), "t3", index, chunks + more)

    loaded, loaded_chunks = load_thread_index(str(tmp_path), "t3")
    assert loaded_chunks == chunks and loaded.get_current_count() == len(chunks)
    assert sorted(p.name for p in (tmp_path / "t3").iterdir()) == ["chunks.json", "index.bin"]


def test_an_upload_cannot_resize_the_index_in_the_middle_of_a_question(monkeypatch):
    # hnswlib must not resize an index while it is being searched: a question's search and get_items
    # must see the index unchanged, even when an upload that outgrows its capacity arrives in between
    monkeypatch.setitem(config.CHUNK_SETTINGS, "CHUNK_SIZE", 10)
    monkeypatch.setitem(config.CHUNK_SETTINGS, "CHUNK_OVERLAP", 0)
    index, chunks = ei.generate_embeddings("abcdefghij" * ei.MIN_CAPACITY)  # exactly at capacity
    assert index.get_current_count() == index.get_max_elements()
    searched, upload_done = threading.Event(), threading.Event()
    seen = {}
    search = ei.find_closest_embeddings_hnsw

    def search_then_let_the_upload_in(*args, **kwargs):
        labels = search(*args, **kwargs)
        seen["before"] = index.get_max_elements()
        searched.set()
        upload_done.wait(timeout=1)  # without a lock the upload resizes the index here
        seen["after"] = index.get_max_elements()
        return labels

    def upload():
        searched.wait(timeout=5)
        ei.add_to_index(index, [vec("klmnopqrst")] * 10)
        upload_done.set()

    monkeypatch.setattr(ei, "find_closest_embeddings_hnsw", search_then_let_the_upload_in)
    uploader = threading.Thread(target=upload)
    uploader.start()
    context = ei.fetch_relevant_data(ei.generate_query_embedding("abcdefghij"), index, chunks)
    uploader.join(timeout=5)
    assert context and seen["before"] == seen["after"] == ei.MIN_CAPACITY
    assert index.get_current_count() == len(chunks) + 10 > index.get_max_elements() // 2




def test_search_falls_back_to_exact_when_the_filtered_walk_fails(small_chunks):
    # hnswlib raises "Cannot return the results in a contiguous 2D array" when a filtered walk finds fewer than k
    class FlakyIndex(hnswlib.Index):
        def knn_query(self, data, k=1, num_threads=-1, filter=None):
            if filter is not None:
                raise RuntimeError("Cannot return the results in a contiguous 2D array. Probably ef or M is too small")
            return super().knn_query(data, k=k, num_threads=num_threads)

    _, chunks = ei.generate_embeddings("cccc dddd " * 12, source="a.txt")
    _, newer = ei.generate_embeddings("aaaa bbbb " * 30, source="b.txt")
    index = FlakyIndex(space="cosine", dim=26)
    index.init_index(max_elements=len(chunks) + len(newer))
    index.add_items(np.array(FakeEmbeddings().embed_documents(chunks + newer)), list(range(len(chunks) + len(newer))))
    context = ei.fetch_relevant_data(ei.generate_query_embedding("aaaa bbbb"), index, chunks)
    assert context.count("[Source: a.txt]") == len(chunks) and "[Source: b.txt]" not in context
