import json
import os
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient


def get_inference_client():
    """Create and return a Hugging Face InferenceClient using the environment token."""
    load_dotenv()
    api_token = os.getenv("HF_API_TOKEN")
    if not api_token:
        raise ValueError("Critical Error: HF_API_TOKEN not found.")

    return InferenceClient(token=api_token)


client = get_inference_client()


def iter_stream_text(stream):
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


st.title("ChatGPT-like clone")

if "hf_model" not in st.session_state:
    st.session_state["hf_model"] = "deepseek-ai/DeepSeek-R1:novita"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["hf_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(iter_stream_text(stream))
    st.session_state.messages.append({"role": "assistant", "content": response})