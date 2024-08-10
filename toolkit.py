from serpapi import GoogleSearch
from langchain_core.tools import tool
import os 

# test multiply tool 
@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiple Tool: Multiply two integers together."""
    print("In Multiply Tool...")
    return first_int * second_int

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
    return answer
