from langchain_core.tools import tool

@tool
def calculator(expression: str) -> int:
    """Calculate an answer for an numerical equation."""
    print("In Calculator Tool...")

    result = eval(expression)
    print(f"Calculator result: {result}")
    return 
