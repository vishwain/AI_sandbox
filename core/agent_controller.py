from core.payload import PayloadBuilder


def run_chat_turn(session, client, model_id, tools=None):
    """Build the payload from *session* history and start a streamed chat completion.

    Returns the raw stream from the HF InferenceClient; the caller pipes it
    through utils.parser.iter_stream_text before rendering.

    Phase-2 note: the agentic/tool-calling loop (run_agentic_loop) will wrap
    this with tool-call detection (utils.parser.parse_tool_calls) and dispatch
    via tools/registry.py, re-invoking the model after appending tool results
    with PayloadBuilder.add_tool_result().
    """
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

    return client.chat.completions.create(stream=True, **builder.build())
