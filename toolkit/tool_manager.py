from toolkit.search import search
from toolkit.calculator import calculator
from config import BASE_PROMPT

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

tools = [calculator, search]
tool_map = {tool.name: tool for tool in tools}

def call_tools(msg: AIMessage, llm) -> Runnable:
    """Simple sequential tool calling helper."""
    global tools, tool_map

    tool_calls = getattr(msg, 'tool_calls', [])

    # If no tools are called, return the LLM response directly
    if not tool_calls:  
        return msg.content
    
    for tool_call in tool_calls:
        tool_call["output"] = tool_map[tool_call["name"]].invoke(tool_call["args"])

    tool_output = tool_call["output"]

    new_prompt = BASE_PROMPT.format(
        context="", 
        question="", 
        relevant_chat_history="",
        last_n="", 
        last_n_messages="",
        tool_call=tool_call
    )

    # Use LLM to generate a neat answer
    final_response = llm.invoke(new_prompt)
    print(f"Final Response: {final_response.content}")
    return final_response.content
