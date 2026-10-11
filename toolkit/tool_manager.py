from langchain_core.messages import AIMessage

from toolkit.calculator import calculator
from toolkit.search import search

tools = [calculator, search]
tool_map = {tool.name: tool for tool in tools}


def call_tools(msg: AIMessage, llm, build_prompt):
    """Run the tools the model asked for, then ask it again with the full original prompt.

    build_prompt(tool_output) must return the complete prompt (question, document
    context, history) with the tool output filled in.
    """
    tool_calls = getattr(msg, "tool_calls", [])

    if not tool_calls:
        return msg.content

    outputs = []
    for tool_call in tool_calls:
        try:
            result = tool_map[tool_call["name"]].invoke(tool_call["args"])
        except Exception as e:
            result = f"Tool error: {e}"
        outputs.append(f"{tool_call['name']}({tool_call['args']}) -> {result}")

    return llm.invoke(build_prompt("\n".join(outputs))).content
