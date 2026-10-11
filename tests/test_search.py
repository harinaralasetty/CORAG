from toolkit import search as search_module


class FakeSerp:
    payload = {}

    def __init__(self, params):
        pass

    def get_dict(self):
        return self.payload


class FakeTavily:
    payload = {}

    def __init__(self, api_key=None):
        pass

    def search(self, query, max_results):
        return self.payload


def run(monkeypatch, provider, payload):
    monkeypatch.setenv("SEARCH_PROVIDER", provider)
    FakeSerp.payload = payload
    FakeTavily.payload = payload
    monkeypatch.setattr(search_module, "GoogleSearch", FakeSerp)
    monkeypatch.setattr(search_module, "TavilyClient", FakeTavily)
    return search_module.search.invoke({"query": "q"})


def test_serpapi_without_results_does_not_crash(monkeypatch):
    assert run(monkeypatch, "serpapi", {"error": "boom"}) == "No results found."


def test_serpapi_joins_snippets(monkeypatch):
    payload = {"organic_results": [{"snippet": "a"}, {"snippet": "b"}]}
    assert run(monkeypatch, "serpapi", payload) == "a\nb"


def test_tavily_joins_results_and_provider_is_case_insensitive(monkeypatch):
    payload = {"results": [{"content": "x"}, {"content": "y"}]}
    assert run(monkeypatch, " Tavily ", payload) == "x\ny"


def test_tavily_empty(monkeypatch):
    assert run(monkeypatch, "tavily", {"results": []}) == "No results found."
