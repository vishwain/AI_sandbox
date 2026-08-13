import os

import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient


@st.cache_resource
def get_inference_client() -> InferenceClient:
    """Create and return a cached Hugging Face InferenceClient using the environment token."""
    load_dotenv()
    api_token = os.getenv("HF_API_TOKEN")
    if not api_token:
        raise ValueError("Critical Error: HF_API_TOKEN not found.")

    return InferenceClient(token=api_token)


def get_available_models() -> dict[str, str]:
    """Return a dict mapping friendly model names to Hugging Face model identifiers.

    Reads the MODELS env var: a comma-separated list of "name: model_id" entries.
    Uses partition() (splits only on the first colon) so model ids that
    themselves contain a colon -- e.g. HF router provider suffixes like
    "deepseek-ai/DeepSeek-R1:novita" -- aren't truncated.
    """
    load_dotenv()
    models = os.getenv("MODELS")
    if not models:
        raise ValueError("Critical Error: MODELS not found.")

    available = {}
    for entry in models.split(","):
        name, sep, model_id = entry.partition(":")
        if not sep:
            continue
        available[name.strip()] = model_id.strip()
    return available


if __name__ == "__main__":
    print(get_available_models())
