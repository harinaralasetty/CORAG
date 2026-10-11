import pytest

from toolkit.calculator import calculator


@pytest.mark.parametrize("expr,expected", [
    ("2+3*4", 14),
    ("(1+2)**3", 27),
    ("-5+2", -3),
    ("10/4", 2.5),
    ("7//2", 3),
    ("7%4", 3),
])
def test_arithmetic(expr, expected):
    assert calculator.invoke({"expression": expr}) == expected


@pytest.mark.parametrize("expr", [
    "__import__('sys').version",
    "open('x')",
    "().__class__",
    "abs(-1)",
    "True+1",
    "2**10000",
    "'a'*5",
    "1;2",
])
def test_rejects_non_arithmetic(expr):
    with pytest.raises((ValueError, SyntaxError)):
        calculator.invoke({"expression": expr})
