"""Ollama models: the real OpenAI client talks to a local stand-in for Ollama's OpenAI-compatible API."""
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import config
from inference import model_router
from inference import ollama_interaction as oi
from inference.llm_error import LLMError


class FakeOllama(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.requests.append((self.path, body))
        reply = json.dumps({
            "id": "1", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "local answer"}}],
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *args):
        pass


@pytest.fixture
def point_client_at(monkeypatch):
    def point(url):
        monkeypatch.setattr(oi, "OLLAMA_BASE_URL", url)
        oi.get_client.cache_clear()
    yield point
    oi.get_client.cache_clear()


def test_ollama_models_are_sent_to_the_local_server_without_the_prefix(point_client_at):
    server = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    FakeOllama.requests.clear()
    try:
        point_client_at(f"http://127.0.0.1:{server.server_port}/v1")
        assert model_router.generate_response("What is in the file?", "ollama/llama3.2") == "local answer"
    finally:
        server.shutdown()
    path, body = FakeOllama.requests[0]
    assert path == "/v1/chat/completions"
    assert body["model"] == "llama3.2"
    assert body["messages"] == [{"role": "user", "content": "What is in the file?"}]


def test_ollama_not_running_raises_a_model_error(point_client_at):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]  # closed again, so nothing listens here
    point_client_at(f"http://127.0.0.1:{port}/v1")
    with pytest.raises(LLMError, match="model request failed"):
        model_router.generate_response("hello", "ollama/llama3.2")


def test_only_gemini_models_use_tools():
    assert model_router.uses_tools("gemini-2.5-flash")
    assert not model_router.uses_tools("claude-haiku-4-5-20251001")
    assert not model_router.uses_tools("ollama/llama3.2")


def test_ollama_models_setting_is_parsed_into_picker_names():
    assert config.ollama_models(" llama3.2, ,qwen2.5:7b ") == ["ollama/llama3.2", "ollama/qwen2.5:7b"]
    assert config.ollama_models("") == []


def test_ollama_is_the_default_when_no_cloud_key_is_set(monkeypatch):
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "")
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(config, "OLLAMA_MODELS", [])
    assert not config.has_any_llm_key()
    monkeypatch.setattr(config, "OLLAMA_MODELS", ["ollama/llama3.2"])
    assert config.has_any_llm_key()
    assert config.default_model() == "ollama/llama3.2"


def test_cloud_keys_win_over_ollama_for_the_default(monkeypatch):
    monkeypatch.setattr(config, "OLLAMA_MODELS", ["ollama/llama3.2"])
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "")
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "a-key")
    assert config.default_model().startswith("claude")
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "g-key")
    assert config.default_model() == config.models_list[0]


def test_configured_ollama_models_are_in_the_picker():
    assert all(model in config.models_list for model in config.OLLAMA_MODELS)
    assert config.models_list[:1] == ["gemini-2.5-flash"]
