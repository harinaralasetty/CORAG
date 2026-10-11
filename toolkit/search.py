import os

from langchain_core.tools import tool
from serpapi import GoogleSearch
from tavily import TavilyClient


@tool
def search(query: str) -> str:
    """Search Tool: Takes a query and searches for relevant information."""
    provider = os.environ.get("SEARCH_PROVIDER", "serpapi")

    if provider == "tavily":
        client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
        response = client.search(query=query, max_results=1)
        results = response.get("results", [])
        if results:
            return results[0].get("content", "")
        return ""
    else:
        params = {
            "engine": "google",
            "q": query,
            "api_key": os.environ.get("SERPER_API_KEY"),
        }
        results = GoogleSearch(params).get_dict()
        return results["organic_results"][0]["snippet"]
