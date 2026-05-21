from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from config import BASE_PROMPT
from toolkit.calculator import calculator
from toolkit.search import search

tools = [calculator, search]
tool_map = {tool.name: tool for tool in tools}


def call_tools(msg: AIMessage, llm) -> Runnable:
    """Simple sequential tool calling helper."""
    tool_calls = getattr(msg, 'tool_calls', [])

    if not tool_calls:
        return msg.content

    for tool_call in tool_calls:
        tool_call["output"] = tool_map[tool_call["name"]].invoke(tool_call["args"])

    new_prompt = BASE_PROMPT.format(
        context="",
        question="",
        relevant_chat_history="",
        last_n="",
        last_n_messages="",
        tool_call=tool_call,
        document_names="none",
    )

    return llm.invoke(new_prompt).content
