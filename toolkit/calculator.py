from langchain_core.tools import tool


@tool
def calculator(expression: str) -> float:
    """Calculate an answer for a numerical equation."""
    result = eval(expression)
    print(f"Calculator result: {result}")
    return result
