from serpapi import GoogleSearch
from langchain_core.tools import tool
import os 

@tool
def search(query: str) -> str:
    """Search Tool: Takes a query and searches for relevant information."""

    params = {
    "engine": "google",
    "q": query,
    "api_key": os.environ.get("SERPER_API_KEY")
    }
    print('In Search Tool... ')
    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results["organic_results"]
    answer = organic_results[0]["snippet"]
    
    print(f"Search result:{answer}")
    return answer
