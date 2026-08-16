import streamlit as st

from core.llm_client import get_available_models


def render_agent_selector():
    """Render the sidebar radio choosing which agent a new chat will use.

    Only affects chats created afterwards, not the active conversation.
    """
    return st.radio("New chat agent", ["General Chat", "Web Research"], horizontal=True)


def render_sidebar(sessions, active_id):
    """Render the left sidebar: agent picker, new-chat button, conversation list, sampling sliders.

    Returns "new_chat", the session_id switched to, or None.
    """
    action = None
    with st.sidebar:
        st.header("Conversations")

        agent_mode = render_agent_selector()

        if st.button("+ Create New Chat", use_container_width=True):
            action = "new_chat"
            st.session_state["pending_agent_mode"] = agent_mode

        st.divider()
        st.subheader("Conversations")

        for session_id, chat_session in reversed(list(sessions.items())):
            label = chat_session.title
            if session_id == active_id:
                label = f"\u27a4 {label}"
            if st.button(label, key=f"conversation_{session_id}", use_container_width=True):
                action = session_id

        st.divider()
        st.header("Sampling")
        active_session = sessions[active_id]
        active_session.temperature = st.slider(
            "Temperature", 0.0, 1.5, active_session.temperature, 0.05, key=f"temp_{active_id}"
        )
        active_session.top_p = st.slider(
            "Top-p", 0.0, 1.0, active_session.top_p, 0.05, key=f"top_p_{active_id}"
        )

    return action


def render_model_selector(active_session):
    """Render the model dropdown for the active conversation.

    Reads/writes active_session.hf_model (a friendly key from
    get_available_models()), so each conversation keeps its own model choice.
    Returns the resolved Hugging Face model id to use for the next request.
    """
    model_options = get_available_models()
    keys = list(model_options.keys())

    if active_session.hf_model not in keys:
        active_session.hf_model = keys[0]

    selected = st.selectbox(
        "Model",
        options=keys,
        index=keys.index(active_session.hf_model),
        key=f"model_{active_session.session_id}",
    )
    active_session.hf_model = selected
    return model_options[selected]


def render_chat_history(messages):
    """Replay history, pairing each assistant tool_calls entry with its tool result."""
    tool_results = {m["tool_call_id"]: m["content"] for m in messages if m["role"] == "tool"}

    for message in messages:
        role = message["role"]
        if role in ("system", "tool"):
            continue

        if role == "assistant" and message.get("tool_calls"):
            with st.chat_message("assistant"):
                for tool_call in message["tool_calls"]:
                    function = tool_call["function"]
                    render_tool_call(
                        function["name"], function["arguments"], tool_results.get(tool_call["id"])
                    )
            if not message.get("content"):
                continue

        with st.chat_message(role):
            if message.get("content"):
                st.markdown(message["content"])
            if message.get("reasoning"):
                render_thinking_expander(message["reasoning"])


def render_tool_call(name, arguments, result=None):
    """Render a single tool invocation as a collapsed expander (call args + result)."""
    with st.expander(f"\U0001f527 {name}({arguments})", expanded=False):
        st.markdown(f"```\n{result}\n```" if result else "_Running..._")


def render_chat_input():
    return st.chat_input("Ask AI")


def render_thinking_expander(reasoning_text):
    with st.expander("Thinking", expanded=False):
        st.markdown(reasoning_text)
