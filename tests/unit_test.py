import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent_controller import run_agentic_loop
from core.payload import PayloadBuilder
from core.session import ChatSession
from tools import registry
from utils.parser import extract_reasoning, iter_stream_text


def test_payload_builder_build_shape():
    builder = PayloadBuilder(model="deepseek-ai/DeepSeek-R1:novita", temperature=0.3, top_p=0.8)
    builder.add_system_prompt("You are helpful.")
    builder.add_user_message("Hi")
    builder.add_assistant_message("Hello!")

    payload = builder.build()

    assert payload["model"] == "deepseek-ai/DeepSeek-R1:novita"
    assert payload["temperature"] == 0.3
    assert payload["top_p"] == 0.8
    assert payload["messages"][0] == {"role": "system", "content": "You are helpful."}
    assert "tools" not in payload


def test_payload_builder_tool_result_round_trip():
    builder = PayloadBuilder(model="m")
    builder.add_user_message("What's 2+2?")
    builder.add_assistant_message(
        None,
        tool_calls=[{"id": "call_1", "function": {"name": "calc", "arguments": "{}"}}],
    )
    builder.add_tool_result("call_1", "calc", "4")

    payload = builder.build()

    assert payload["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "calc",
        "content": "4",
    }


def test_payload_builder_truncate_context_keeps_system_and_recent():
    builder = PayloadBuilder(model="m")
    builder.add_system_prompt("sys")
    for _ in range(20):
        builder.add_user_message("x" * 100)

    builder.truncate_context(max_tokens=50)

    assert builder.messages[0]["role"] == "system"
    assert len(builder.messages) < 21


def test_chat_session_add_message_and_history():
    session = ChatSession(system_prompt="sys", hf_model="deepseekR1")
    session.add_message("user", "hello")
    session.add_message("assistant", "hi there", reasoning="thinking...")

    history = session.get_history()

    assert history[0] == {"role": "system", "content": "sys"}
    assert history[1] == {"role": "user", "content": "hello"}
    assert history[2]["reasoning"] == "thinking..."


def test_chat_session_maybe_set_title_from_only_sets_once():
    session = ChatSession(system_prompt="sys", hf_model="deepseekR1")
    session.maybe_set_title_from("What is the capital of France?")
    first_title = session.title

    session.maybe_set_title_from("A different message")

    assert session.title == first_title


def test_iter_stream_text_strips_think_tags_and_collects_reasoning():
    chunks = ["<thi", "nk>internal reasoning</thin", "k>visible answer"]
    reasoning_sink = []

    visible = "".join(iter_stream_text(iter(chunks), reasoning_sink))

    assert visible == "visible answer"
    assert "".join(reasoning_sink) == "internal reasoning"


def test_extract_reasoning_plain_string():
    visible, reasoning = extract_reasoning("<think>hmm</think>final answer")

    assert visible == "final answer"
    assert reasoning == "hmm"


def test_get_available_models_partition_bug_regression(monkeypatch):
    from core import llm_client

    monkeypatch.setenv(
        "MODELS", "deepseekR1: deepseek-ai/DeepSeek-R1:novita, Qwen3: qwen/Qwen-3.0-Chat"
    )

    models = llm_client.get_available_models()

    assert models["deepseekR1"] == "deepseek-ai/DeepSeek-R1:novita"
    assert models["Qwen3"] == "qwen/Qwen-3.0-Chat"


def test_registry_dispatch_success():
    registry.register_tool("_test_echo", {"type": "function", "function": {"name": "_test_echo"}}, lambda text: text.upper())

    result = registry.dispatch("_test_echo", '{"text": "hi"}')

    assert result == "HI"


def test_registry_dispatch_unknown_tool_returns_error_string():
    result = registry.dispatch("_does_not_exist", "{}")

    assert result.startswith("Error:")


def test_registry_dispatch_handler_exception_returns_error_string():
    def _boom():
        raise ValueError("kaboom")

    registry.register_tool("_test_boom", {"type": "function", "function": {"name": "_test_boom"}}, _boom)

    result = registry.dispatch("_test_boom", "{}")

    assert result.startswith("Error:")
    assert "kaboom" in result


def test_web_research_agent_exposes_system_prompt_and_tools():
    from agents import web_research_agent

    assert web_research_agent.SYSTEM_PROMPT.strip() != ""
    assert len(web_research_agent.TOOLS) == 2
    tool_names = {tool["function"]["name"] for tool in web_research_agent.TOOLS}
    assert tool_names == {"web_search", "fetch_page"}


def test_chat_session_add_message_tool_call_round_trip():
    session = ChatSession(system_prompt="sys", hf_model="m")
    tool_calls = [{"id": "call_1", "function": {"name": "web_search", "arguments": "{}"}}]

    session.add_message("assistant", None, tool_calls=tool_calls)
    session.add_message("tool", "result text", tool_call_id="call_1", name="web_search")

    history = session.get_history()

    assert history[1]["tool_calls"] == tool_calls
    assert history[2] == {
        "role": "tool",
        "content": "result text",
        "tool_call_id": "call_1",
        "name": "web_search",
    }


class _FakeToolCallFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = _FakeToolCallFunction(name, arguments)


class _FakeMessage:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, message, finish_reason):
        self.message = message
        self.finish_reason = finish_reason


class _FakeResponse:
    def __init__(self, message, finish_reason):
        self.choices = [_FakeChoice(message, finish_reason)]


class _FakeClient:
    """Stub client: first call returns a tool_calls response, then a final answer."""

    def __init__(self):
        self.calls = []
        self.chat = self
        self.completions = self

    def create(self, stream, **kwargs):
        self.calls.append({"stream": stream, **kwargs})
        if len(self.calls) == 1:
            tool_calls = [_FakeToolCall("call_1", "_test_echo", '{"text": "hi"}')]
            return _FakeResponse(_FakeMessage(None, tool_calls), "tool_calls")
        if stream:
            return iter(["final answer"])
        return _FakeResponse(_FakeMessage("final answer"), "stop")


def test_run_agentic_loop_dispatches_tool_once_then_streams_final_answer():
    registry.register_tool("_test_echo", {"type": "function", "function": {"name": "_test_echo"}}, lambda text: text.upper())
    session = ChatSession(system_prompt="sys", hf_model="m")
    session.add_message("user", "search something")
    client = _FakeClient()

    stream = run_agentic_loop(session, client, "m", tools=[], max_iterations=3)

    assert list(stream) == ["final answer"]
    assert len(client.calls) == 3
    assert client.calls[0]["stream"] is False
    assert client.calls[1]["stream"] is False
    assert client.calls[2]["stream"] is True
    tool_messages = [m for m in session.get_history() if m["role"] == "tool"]
    assert tool_messages == [
        {"role": "tool", "content": "HI", "tool_call_id": "call_1", "name": "_test_echo"}
    ]
