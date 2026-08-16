import streamlit as st

from agents import web_research_agent
from core.agent_controller import run_agentic_loop, run_chat_turn
from core.llm_client import get_available_models, get_inference_client
from core.session import ChatSession
from ui.layout import (
    render_chat_history,
    render_chat_input,
    render_model_selector,
    render_sidebar,
    render_thinking_expander,
    render_tool_call,
)
from utils.parser import iter_stream_text

SYSTEM_PROMPT = "You are a helpful assistant."


def _create_new_session():
    default_model = next(iter(get_available_models()))
    agent_mode = st.session_state.pop("pending_agent_mode", "General Chat")
    if agent_mode == "Web Research":
        new_session = ChatSession(
            system_prompt=web_research_agent.SYSTEM_PROMPT,
            hf_model=default_model,
            tools=web_research_agent.TOOLS,
        )
    else:
        new_session = ChatSession(system_prompt=SYSTEM_PROMPT, hf_model=default_model)
    st.session_state.sessions[new_session.session_id] = new_session
    st.session_state.active_id = new_session.session_id


def _ensure_state():
    if "sessions" not in st.session_state:
        st.session_state.sessions = {}
    if "active_id" not in st.session_state:
        st.session_state.active_id = None
    if not st.session_state.sessions:
        _create_new_session()


def main():
    st.set_page_config(page_title="AI Sandbox", page_icon="\U0001f916")
    st.title("AI Sandbox")

    try:
        _ensure_state()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
        return

    action = render_sidebar(st.session_state.sessions, st.session_state.active_id)
    if action == "new_chat":
        _create_new_session()
        st.rerun()
    elif action is not None:
        st.session_state.active_id = action
        st.rerun()

    active_session = st.session_state.sessions[st.session_state.active_id]

    model_id = render_model_selector(active_session)
    render_chat_history(active_session.get_history())

    prompt = render_chat_input()
    if prompt:
        active_session.add_message("user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            client = get_inference_client()
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
            return

        with st.chat_message("assistant"):
            if active_session.tools:
                stream = run_agentic_loop(
                    active_session,
                    client,
                    model_id,
                    active_session.tools,
                    on_tool_call=render_tool_call,
                )
            else:
                stream = run_chat_turn(active_session, client, model_id)
            reasoning_sink = []
            response_text = st.write_stream(iter_stream_text(stream, reasoning_sink))
            reasoning_text = "".join(reasoning_sink)
            if reasoning_text:
                render_thinking_expander(reasoning_text)

        active_session.add_message("assistant", response_text, reasoning=reasoning_text or None)
        active_session.maybe_set_title_from(prompt)
        st.rerun()


if __name__ == "__main__":
    main()
