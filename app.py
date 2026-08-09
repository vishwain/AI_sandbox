import json
import os
import streamlit as st
from dotenv import load_dotenv
import core.session as session
from ui.layout import build_main_layout
from huggingface_hub import InferenceClient

load_dotenv()
DEBUG_REASONING = os.getenv("DEBUG_REASONING", "").lower() in ("true", "1", "yes")

global current_session


def get_current_session() -> session.Session:
    """Get the current session, creating a new one if it doesn't exist."""
    global current_session
    if current_session is None:
        current_session = session.Session()
    return current_session


def set_current_session(new_session: session.Session):
    """Set the current session."""
    global current_session
    current_session = new_session


def iter_stream_text(stream):
    """Yield plain text from streamed chunks, including raw JSON strings.

    Reasoning content wrapped in <think>... think tags is stripped by default.
    Set the environment variable DEBUG_REASONING=true to emit it.
    """
    yield from _process_think_tags(_extract_text(stream))


def _extract_text(stream):
    """Yield plain text from streamed chunks, including raw JSON strings."""
    for chunk in stream:
        if isinstance(chunk, str):
            try:
                payload = json.loads(chunk)
            except json.JSONDecodeError:
                yield chunk
                continue

            if isinstance(payload, dict):
                message = payload.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        yield content
                        continue

                choices = payload.get("choices", [])
                if choices and isinstance(choices[0], dict):
                    delta = choices[0].get("delta", {})
                    if isinstance(delta, dict):
                        content = delta.get("content")
                        if isinstance(content, str):
                            yield content
                            continue

                if isinstance(payload.get("content"), str):
                    yield payload["content"]
                continue

        if isinstance(chunk, (bytes, bytearray)):
            try:
                text = chunk.decode("utf-8")
            except UnicodeDecodeError:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                yield text
                continue

            if isinstance(payload, dict):
                message = payload.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        yield content
                        continue

                choices = payload.get("choices", [])
                if choices and isinstance(choices[0], dict):
                    delta = choices[0].get("delta", {})
                    if isinstance(delta, dict):
                        content = delta.get("content")
                        if isinstance(content, str):
                            yield content
                            continue

                if isinstance(payload.get("content"), str):
                    yield payload["content"]
            continue

        if hasattr(chunk, "choices") and getattr(chunk, "choices", None):
            choice = chunk.choices[0]
            delta = getattr(choice, "delta", None)
            if delta and getattr(delta, "content", None):
                yield delta.content
                continue

        if hasattr(chunk, "delta") and getattr(chunk, "delta", None):
            content = getattr(chunk.delta, "content", None)
            if isinstance(content, str):
                yield content
                continue

        if isinstance(chunk, dict):
            message = chunk.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    yield content
                    continue

            if isinstance(chunk.get("content"), str):
                yield chunk["content"]


def _process_think_tags(text_stream):
    """Yield text from *text_stream*, handling  think reasoning blocks.

    Text outside  think tags is always emitted. When DEBUG_REASONING is
    enabled, the content of each  think block is emitted; otherwise it is
    discarded. Split tags across streamed chunks are handled correctly.
    """
    buffer = ""
    in_think = False
    think_start = "<think>"
    think_end = "</think>"

    for text in text_stream:
        buffer += text
        while buffer:
            if not in_think:
                start = buffer.find(think_start)
                if start == -1:
                    # Keep a tail long enough to not split a partial <think> tag.
                    if len(buffer) > len(think_start):
                        yield buffer[:-len(think_start)]
                        buffer = buffer[-len(think_start):]
                    break
                if start > 0:
                    yield buffer[:start]
                buffer = buffer[start + len(think_start):]
                in_think = True
                continue

            end = buffer.find(think_end)
            if end == -1:
                if DEBUG_REASONING:
                    # Keep a tail long enough to not split a partial  think end tag.
                    if len(buffer) > len(think_end):
                        yield buffer[:-len(think_end)]
                        buffer = buffer[-len(think_end):]
                    # else: buffer is small, keep it for the next chunk.
                else:
                    # Discard reasoning text, but preserve a tail that may hold
                    # a partial  think end tag.
                    buffer = buffer[-len(think_end):] if len(buffer) > len(think_end) else buffer
                break

            if DEBUG_REASONING:
                yield buffer[:end]
            buffer = buffer[end + len(think_end):]
            in_think = False

    # Flush any remaining plain text. If we are still inside a think block,
    # only emit the remainder when DEBUG_REASONING is enabled.
    if buffer and not in_think:
        yield buffer
    elif buffer and in_think and DEBUG_REASONING:
        yield buffer


if __name__ == "__main__":

    page_title = "ChatGPT-like clone"    # later will be changed to current conversation topic
    st.set_page_config(page_title=page_title, page_icon="🤖")
    st.title(page_title)

    build_main_layout()

    # client = get_inference_client()

    if "hf_model" not in st.session_state:
        st.session_state["hf_model"] = "deepseek-ai/DeepSeek-R1:novita"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "conversations" not in st.session_state:
        st.session_state.conversations = []

    if "current_title" not in st.session_state:
        st.session_state.current_title = "Chat Session 1"

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    # if prompt := st.chat_input("What is up?"):
    #     st.session_state.messages.append({"role": "user", "content": prompt})
    #     with st.chat_message("user"):
    #         st.markdown(prompt)

        # with st.chat_message("assistant"):
        #     stream = client.chat.completions.create(
        #         model=st.session_state["hf_model"],
        #         messages=[
        #             {"role": m["role"], "content": m["content"]}
        #             for m in st.session_state.messages
        #         ],
        #         stream=True,
        #     )
        #     response = st.write_stream(iter_stream_text(stream))
        # st.session_state.messages.append({"role": "assistant", "content": response})