from toolkit.tool_manager import call_tools


class Msg:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class EchoLLM:
    def invoke(self, prompt):
        return Msg(content=prompt)


def build_prompt(tool_output=""):
    return f"QUESTION: what is 2+2?\nCONTEXT: doc text\nTOOLS: {tool_output}"


def test_no_tool_calls_returns_content():
    assert call_tools(Msg(content="hi"), EchoLLM(), build_prompt) == "hi"


def test_tool_result_is_added_to_the_full_prompt():
    msg = Msg(tool_calls=[{"name": "calculator", "args": {"expression": "2+2"}}])
    out = call_tools(msg, EchoLLM(), build_prompt)
    assert "what is 2+2" in out and "doc text" in out
    assert "4" in out


def test_all_tool_calls_are_kept():
    msg = Msg(tool_calls=[
        {"name": "calculator", "args": {"expression": "1+1"}},
        {"name": "calculator", "args": {"expression": "3*3"}},
    ])
    out = call_tools(msg, EchoLLM(), build_prompt)
    assert "2" in out and "9" in out


def test_tool_errors_are_reported_not_raised():
    msg = Msg(tool_calls=[
        {"name": "nope", "args": {}},
        {"name": "calculator", "args": {"expression": "__import__('os')"}},
    ])
    out = call_tools(msg, EchoLLM(), build_prompt)
    assert out.count("Tool error") == 2
