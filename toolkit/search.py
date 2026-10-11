import os

from langchain_core.tools import tool
from serpapi import GoogleSearch
from tavily import TavilyClient

MAX_RESULTS = 3
NO_RESULTS = "No results found."


def _tavily_search(query: str) -> list[str]:
    client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
    response = client.search(query=query, max_results=MAX_RESULTS)
    return [r.get("content", "") for r in response.get("results", [])]


def _serpapi_search(query: str) -> list[str]:
    params = {
        "engine": "google",
        "q": query,
        "api_key": os.environ.get("SERPER_API_KEY"),
    }
    results = GoogleSearch(params).get_dict()
    return [r.get("snippet", "") for r in results.get("organic_results", [])[:MAX_RESULTS]]


@tool
def search(query: str) -> str:
    """Search Tool: Takes a query and searches for relevant information."""
    provider = os.environ.get("SEARCH_PROVIDER", "serpapi").strip().lower()
    snippets = _tavily_search(query) if provider == "tavily" else _serpapi_search(query)
    return "\n".join(s for s in snippets if s) or NO_RESULTS
