import streamlit as st
import core.session as session
from core.llm_client import get_available_models


def build_main_layout():
    '''
    Build the main layout of the app.
    '''
    add_chat_input()
    add_side_bar()
    add_model_selection_dropdown()


def add_chat_input():
    '''
    Add a chat input box to the main layout.
    '''
    prompt = st.chat_input("Ask AI")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            from core.llm_client import get_inference_client
            client = get_inference_client()
            stream = client.chat.completions.create(
                model=st.session_state["hf_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
            )


def add_side_bar():
    '''
    Add a sidebar on the left to the app.
    Consists of list of user's previous conversations and a button(+) to start a new conversation.
    On clicking the button, callback function `add_new_conversation_callback` is triggered.
    '''
    with st.sidebar:
        st.header("Conversations")

        if "conversations" not in st.session_state:
            st.session_state.conversations = []

        # Render the list of previous conversations
        for idx, conv in enumerate(st.session_state.conversations):
            title = conv.get("title", f"Chat Session {idx + 1}")
            st.button(title, key=f"conversation_{idx}", on_click=activate_conversation, args=(idx,))

        # Button to start a new conversation
        st.button(
            "+",
            key="new_conversation_button",
            on_click=add_new_conversation_callback,
            use_container_width=True,
        )


def add_new_conversation_callback():
    '''
    Create a new conversation and reset the session state.
    '''
    if "conversations" not in st.session_state:
        st.session_state.conversations = []

    # Save the current conversation before starting a new one
    if "messages" in st.session_state and st.session_state.messages:
        current_title = st.session_state.get(
            "current_title",
            f"Chat Session {len(st.session_state.conversations) + 1}",
        )
        st.session_state.conversations.append(
            {"title": current_title, "messages": list(st.session_state.messages)}
        )

    # Reset session state for the new conversation
    st.session_state.messages = []
    st.session_state.current_title = f"Chat Session {len(st.session_state.conversations) + 1}"
    st.session_state.current_session = session.Session()


def activate_conversation(conversation_index):
    '''
    Activate a conversation from the list of previous conversations.
    '''
    if "conversations" not in st.session_state:
        st.session_state.conversations = []

    selected_conversation = st.session_state.conversations[conversation_index]
    st.session_state.messages = selected_conversation.get("messages", [])
    st.session_state.current_title = selected_conversation.get(
        "title", f"Chat Session {conversation_index + 1}"
    )
    st.rerun()


def add_model_selection_dropdown():
    '''
    Add a dropdown to the sidebar for model selection. Will be populated with available models from the .env file.
    If only one conversation is available, default model (Qwen 3) will be selected. For multiple conversations, the model used in the last conversation will be selected.
    '''
    with st.sidebar:
        st.header("Model Selection")
        model_options = get_available_models()
        selected_model = st.selectbox(
            "Select a model:",
            options=list(model_options.keys()),
            index=list(model_options.keys()).index(st.session_state.get("hf_model", "deepseekR1")),
            key="hf_model_dropdown",
            on_change=model_selection_callback,
        )

def model_selection_callback():
    '''
    Callback function to handle model selection from the dropdown.
    '''
    selected_model = st.session_state.get("hf_model", "deepseek-ai/DeepSeek-R1:novita")
    st.session_state.hf_model = selected_model