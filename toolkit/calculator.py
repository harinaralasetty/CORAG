import ast
import operator

from langchain_core.tools import tool

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

MAX_POW_BASE = 1_000_000
MAX_POW_EXPONENT = 1000


def safe_eval(expression: str):
    """Evaluate plain arithmetic only (numbers, + - * / // % ** and parentheses)."""

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Pow) and (abs(left) > MAX_POW_BASE or abs(right) > MAX_POW_EXPONENT):
                raise ValueError("Exponent or base too large")
            return _BINARY_OPS[type(node.op)](left, right)
        raise ValueError("Only plain arithmetic is supported")

    return _eval(ast.parse(expression.strip(), mode="eval"))


@tool
def calculator(expression: str) -> float:
    """Calculate an answer for a numerical equation using + - * / // % ** and parentheses."""
    return safe_eval(expression)
