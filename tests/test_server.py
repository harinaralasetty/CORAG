"""Drive the real NiceGUI app: upload files, ask a question, restart, ask again."""
import asyncio
import json
import os
import time

import numpy as np
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from nicegui import ui
from nicegui.elements.upload_files import SmallFileUpload
from nicegui.testing import User
from nicegui.testing.general_fixtures import get_path_to_main_file
from nicegui.testing.user_simulation import user_simulation

import config
from inference import gemini_interaction as gi
from inference import inference_manager as im
from inference import model_router
from preprocessing import prompt_processor as pp
from retrieval import embeddings_indexing as ei

THREADS_FILE = os.path.join(config.DATA_DIR, "chat_threads_info.json")


def vec(text):
    v = np.zeros(26, dtype=np.float32) + 1e-3
    for ch in text.lower():
        if "a" <= ch <= "z":
            v[ord(ch) - 97] += 1
    return (v / np.linalg.norm(v)).tolist()


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [vec(t) for t in texts]


class FakeLLM:
    """Answers with the full prompt it was given, so tests can inspect the retrieved context."""

    def bind_tools(self, tools):
        return RunnableLambda(lambda prompt: AIMessage(content=f"REPLY|{prompt}"))


def make_pdf(text):
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = b"%PDF-1.4\n", []
    for i, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (len(objs) + 1, xref)
    return out


def read_threads():
    if not os.path.isfile(THREADS_FILE):
        return {}
    with open(THREADS_FILE, encoding="utf-8") as f:
        return json.load(f)


async def wait_for(condition, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return
        await asyncio.sleep(0.2)
    raise AssertionError("timed out waiting for the app")


def only_thread():
    threads = read_threads()
    assert len(threads) == 1
    return next(iter(threads.items()))


def install_fakes(monkeypatch):
    monkeypatch.setattr(ei, "get_embeddings_model", lambda provider: FakeEmbeddings())
    monkeypatch.setattr(im, "get_model_instance", lambda model: FakeLLM())
    monkeypatch.setattr(im, "rephrase_prompt", lambda question, theme, model: question)
    monkeypatch.setattr(pp, "generate_response", lambda prompt, model: "Test Chat")
    monkeypatch.setitem(config.CHUNK_SETTINGS, "CHUNK_SIZE", 60)
    monkeypatch.setitem(config.CHUNK_SETTINGS, "CHUNK_OVERLAP", 10)


async def upload(user, name, text):
    element = user.find(ui.upload).elements.pop()
    file = SmallFileUpload(name=name, content_type="application/pdf", _data=make_pdf(text))
    await element.handle_uploads([file])


async def ask(user, question, answers_wanted):
    user.find(ui.input).type(question).trigger("keydown.enter")

    def answered():
        threads = read_threads()
        if not threads:
            return False
        messages = next(iter(threads.values())).get("messages", [])
        return len(messages) >= answers_wanted and messages[-1].get("answer", "").startswith("REPLY|")

    await wait_for(answered)
    return next(iter(read_threads().values()))["messages"][-1]["answer"]


async def test_two_files_in_one_chat_are_both_searchable(user: User, monkeypatch):
    install_fakes(monkeypatch)
    await user.open("/")
    await user.should_see("CORAG")

    await upload(user, "apples.pdf", "Apples are red and grow on trees in the orchard.")
    await wait_for(lambda: any(t.get("processed_files") for t in read_threads().values()))
    await upload(user, "zebras.pdf", "Zebras have black and white stripes on the savanna.")
    await wait_for(lambda: len(next(iter(read_threads().values()))["processed_files"]) == 2)

    thread_id, thread = only_thread()
    assert [f["name"] for f in thread["processed_files"]] == ["apples.pdf", "zebras.pdf"]
    assert os.path.isfile(os.path.join(config.DATA_DIR, thread_id, "index.bin"))

    answer = await ask(user, "Tell me about zebras and apples", answers_wanted=1)
    assert "[Source: apples.pdf]" in answer and "[Source: zebras.pdf]" in answer
    assert "Tell me about zebras and apples" in answer
    assert "_thinking" not in answer  # the in-flight placeholder must not leak into the prompt
    assert read_threads()[thread_id]["name"] == "Test Chat"


async def test_chat_and_documents_survive_a_restart(user: User, monkeypatch):
    install_fakes(monkeypatch)
    thread_id, before = only_thread()  # written by the previous test, i.e. before this "restart"

    await user.open("/")
    await user.should_see("Test Chat")  # chat list is loaded from disk
    user.find(content="Test Chat").click()
    await user.should_see("zebras.pdf")  # documents of the chat are listed again

    answer = await ask(user, "What about zebras now", answers_wanted=len(before["messages"]) + 1)
    assert "[Source: zebras.pdf]" in answer and "[Source: apples.pdf]" in answer


async def test_two_questions_in_flight_each_keep_their_own_answer(user: User, monkeypatch):
    install_fakes(monkeypatch)

    class SlowAgent:
        def run(self, user_input, **kwargs):
            time.sleep(1.0 if user_input == "first question" else 0.1)
            return f"REPLY|{user_input}"

    monkeypatch.setattr(im, "initialize_agent", lambda *args, **kwargs: SlowAgent())
    await user.open("/")
    user.find(ui.input).type("first question").trigger("keydown.enter")
    await asyncio.sleep(0.2)
    user.find(ui.input).type("second question").trigger("keydown.enter")

    def messages():
        for thread in read_threads().values():
            if any(m["user"] == "first question" for m in thread["messages"]):
                return thread["messages"]
        return []

    await wait_for(lambda: any(m["answer"] == "REPLY|first question" for m in messages()))
    assert [(m["user"], m["answer"]) for m in messages()] == [
        ("first question", "REPLY|first question"),
        ("second question", "REPLY|second question"),
    ]


async def test_a_failed_answer_is_saved_to_disk(user: User, monkeypatch):
    install_fakes(monkeypatch)

    class FailingAgent:
        def run(self, user_input, **kwargs):
            raise RuntimeError("model unavailable")

    monkeypatch.setattr(im, "initialize_agent", lambda *args, **kwargs: FailingAgent())
    await user.open("/")
    user.find(ui.input).type("doomed question").trigger("keydown.enter")

    def saved_answer():
        for thread in read_threads().values():
            for message in thread["messages"]:
                if message["user"] == "doomed question":
                    return message["answer"]
        return ""

    await wait_for(lambda: saved_answer().startswith("**Error:**"))  # a restart would otherwise show "_thinking…_"
    assert "model unavailable" in saved_answer()


async def test_a_chat_whose_index_cannot_be_loaded_asks_for_a_reupload(request, monkeypatch, caplog):
    install_fakes(monkeypatch)
    tid = "broken-index-chat"
    folder = os.path.join(config.DATA_DIR, tid)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "index.bin"), "wb") as f:
        f.write(b"\0" * 16)
    with open(os.path.join(folder, "chunks.json"), "w", encoding="utf-8") as f:
        f.write('{"dim": 26, "te')  # cut off mid-write
    threads = read_threads()
    threads[tid] = {
        "name": "Broken Index Chat",
        "messages": [],
        "processed_files": [{"name": "lost.pdf", "type": "application/pdf"}],
        "document_theme": "Lost",
        "embedding_provider": "google",
        "created_at": "2026-01-01T00:00:00",
        "last_updated_at": "2026-01-01T00:00:00",
    }
    with open(THREADS_FILE, "w", encoding="utf-8") as f:
        json.dump(threads, f)

    async with user_simulation(main_file=get_path_to_main_file(request)) as user:  # app starts after the damage
        await user.open("/")
        user.find(content="Broken Index Chat").click()
        await user.should_see("No files uploaded yet.")
        assert user.notify.contains("must be re-uploaded: lost.pdf")

    assert read_threads()[tid]["processed_files"] == []
    assert tid in caplog.text


async def test_the_same_file_uploaded_twice_at_once_is_processed_once(user: User, monkeypatch):
    install_fakes(monkeypatch)
    await user.open("/")
    element = user.find(ui.upload).elements.pop()
    data = make_pdf("Twins share one birthday and one cake.")
    await element.handle_uploads([
        SmallFileUpload(name="twins.pdf", content_type="application/pdf", _data=data),
        SmallFileUpload(name="twins.pdf", content_type="application/pdf", _data=data),
    ])

    def outcomes():
        return [m for m in user.notify.messages if m in ("Processed twins.pdf", "twins.pdf already processed")]

    await wait_for(lambda: len(outcomes()) == 2)
    assert sorted(outcomes()) == ["Processed twins.pdf", "twins.pdf already processed"]
    files = [t["processed_files"] for t in read_threads().values()
             if any(f["name"] == "twins.pdf" for f in t["processed_files"])]
    assert files == [[{"name": "twins.pdf", "type": "application/pdf"}]]


async def test_a_question_asked_while_another_is_answered_does_not_see_its_placeholder(user: User, monkeypatch):
    install_fakes(monkeypatch)
    histories = {}

    class SlowAgent:
        def run(self, user_input, chat_history=None, **kwargs):
            histories[user_input] = list(chat_history or [])
            time.sleep(1.0 if user_input == "slow question" else 0.1)
            return f"REPLY|{user_input}"

    monkeypatch.setattr(im, "initialize_agent", lambda *args, **kwargs: SlowAgent())
    await user.open("/")
    user.find(ui.input).type("slow question").trigger("keydown.enter")
    await asyncio.sleep(0.2)
    user.find(ui.input).type("quick question").trigger("keydown.enter")
    await wait_for(lambda: "quick question" in histories)
    assert histories["quick question"] == []  # the slow question has no answer yet

    await wait_for(lambda: any(m.get("answer") == "REPLY|slow question"
                               for t in read_threads().values() for m in t["messages"]))
    user.find(ui.input).type("later question").trigger("keydown.enter")
    await wait_for(lambda: "later question" in histories)
    assert [h.split("\n")[0] for h in histories["later question"]] == [
        "User: slow question ", "User: quick question "]
    assert not any("_thinking" in h for h in histories["later question"])


class BrokenModel:
    def invoke(self, prompt):
        raise RuntimeError(f"quota exceeded for key {os.environ['GOOGLE_API_KEY']}")


class BrokenToolsLLM:
    def bind_tools(self, tools):
        return RunnableLambda(lambda prompt: BrokenModel().invoke(prompt))


def finished_answer(question):
    for thread in read_threads().values():
        for message in thread["messages"]:
            if message["user"] == question and not message["answer"].startswith("_thinking"):
                return thread, message["answer"]
    return None, ""


async def test_a_failed_chat_naming_keeps_a_neutral_name(user: User, monkeypatch):
    install_fakes(monkeypatch)
    monkeypatch.setattr(pp, "generate_response", model_router.generate_response)
    monkeypatch.setattr(gi, "get_model_instance", lambda model: BrokenModel())  # only naming reaches it
    await user.open("/")
    user.find(ui.input).type("name this chat").trigger("keydown.enter")
    await wait_for(lambda: finished_answer("name this chat")[0])
    thread, answer = finished_answer("name this chat")
    assert answer.startswith("REPLY|")
    assert thread["name"] == "Temporary Thread"


async def test_a_failed_model_call_shows_an_error_not_a_reply(user: User, monkeypatch):
    install_fakes(monkeypatch)
    monkeypatch.setattr(im, "get_model_instance", lambda model: BrokenToolsLLM())
    monkeypatch.setattr(gi, "get_model_instance", lambda model: BrokenModel())  # the plain-prompt fallback
    await user.open("/")
    user.find(ui.input).type("unlucky question").trigger("keydown.enter")
    await wait_for(lambda: finished_answer("unlucky question")[0])
    answer = finished_answer("unlucky question")[1]
    assert answer.startswith("**Error:**") and "quota exceeded" in answer
    assert os.environ["GOOGLE_API_KEY"] not in answer


def thread_with(*names):
    """The saved chat whose documents include all the given file names (earlier tests leave chats behind)."""
    for thread in read_threads().values():
        if set(names) <= {f["name"] for f in thread.get("processed_files", [])}:
            return thread
    return None


async def test_text_and_markdown_files_are_searchable(user: User, monkeypatch):
    install_fakes(monkeypatch)
    await user.open("/")
    element = user.find(ui.upload).elements.pop()
    await element.handle_uploads([
        SmallFileUpload(name="notes.md", content_type="text/markdown",
                        _data="\ufeff# Notes\n\nKoalas sleep in eucalyptus trees.".encode("utf-8")),
    ])
    await wait_for(lambda: thread_with("notes.md"))
    await element.handle_uploads([
        SmallFileUpload(name="list.txt", content_type="text/plain", _data=b"Otters hold hands while sleeping."),
    ])
    await wait_for(lambda: thread_with("notes.md", "list.txt"))

    question = "Where do koalas and otters sleep?"
    user.find(ui.input).type(question).trigger("keydown.enter")

    def answer():
        return next((m["answer"] for m in thread_with("notes.md", "list.txt")["messages"]
                     if m["user"] == question and m["answer"].startswith("REPLY|")), None)

    await wait_for(answer)
    assert "[Source: notes.md]" in answer() and "eucalyptus" in answer()
    assert "[Source: list.txt]" in answer() and "Otters" in answer()
