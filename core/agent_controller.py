from core.payload import PayloadBuilder
from tools import registry

MAX_TOOL_ITERATIONS = 5


def _build_payload(session, model_id, tools):
    """Assemble a PayloadBuilder from *session* history, ready for the InferenceClient."""
    builder = PayloadBuilder(
        model=model_id,
        tools=tools,
        temperature=session.temperature,
        top_p=session.top_p,
    )

    for message in session.get_history():
        role = message["role"]
        content = message["content"]
        if role == "system":
            builder.add_system_prompt(content)
        elif role == "user":
            builder.add_user_message(content)
        elif role == "assistant":
            builder.add_assistant_message(content, tool_calls=message.get("tool_calls"))
        elif role == "tool":
            builder.add_tool_result(message["tool_call_id"], message["name"], content)

    return builder


def run_chat_turn(session, client, model_id, tools=None):
    """Build the payload from *session* history and start a streamed chat completion.

    Returns the raw stream from the HF InferenceClient; the caller pipes it
    through utils.parser.iter_stream_text before rendering.
    """
    builder = _build_payload(session, model_id, tools)
    return client.chat.completions.create(stream=True, **builder.build())


def run_agentic_loop(session, client, model_id, tools, max_iterations=MAX_TOOL_ITERATIONS, on_tool_call=None):
    """Resolve tool calls non-streamed, then return a streamed final answer.

    Tool-call detection needs a complete response (finish_reason, full
    tool_calls array), so each resolution round is a non-streaming request.
    Once the model stops requesting tools (or max_iterations is hit), the
    final answer is fetched via the normal streamed run_chat_turn so the UI
    streaming path (st.write_stream) is unaffected.

    If given, *on_tool_call(name, arguments, result)* is invoked right after
    each dispatch so the caller can render the call live as it happens.
    """
    for _ in range(max_iterations):
        builder = _build_payload(session, model_id, tools)
        response = client.chat.completions.create(stream=False, **builder.build())
        message = response.choices[0].message

        if response.choices[0].finish_reason != "tool_calls" or not message.tool_calls:
            break

        session.add_message("assistant", message.content, tool_calls=message.tool_calls)
        for tool_call in message.tool_calls:
            result = registry.dispatch(tool_call.function.name, tool_call.function.arguments)
            session.add_message(
                "tool", result, tool_call_id=tool_call.id, name=tool_call.function.name
            )
            if on_tool_call:
                on_tool_call(tool_call.function.name, tool_call.function.arguments, result)

    return run_chat_turn(session, client, model_id, tools)

