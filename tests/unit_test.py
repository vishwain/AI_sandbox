import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.payload import PayloadBuilder
from core.session import ChatSession
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
