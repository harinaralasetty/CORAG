import os

from langchain_core.tools import tool
from serpapi import GoogleSearch


@tool
def search(query: str) -> str:
    """Search Tool: Takes a query and searches for relevant information."""
    params = {
        "engine": "google",
        "q": query,
        "api_key": os.environ.get("SERPER_API_KEY"),
    }
    results = GoogleSearch(params).get_dict()
    return results["organic_results"][0]["snippet"]
