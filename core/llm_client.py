from dotenv import load_dotenv
import os
from huggingface_hub import InferenceClient

def get_inference_client() -> InferenceClient:
    """Create and return a Hugging Face InferenceClient using the environment token."""
    api_token = os.getenv("HF_API_TOKEN")
    if not api_token:
        raise ValueError("Critical Error: HF_API_TOKEN not found.")

    return InferenceClient(token=api_token)

def get_available_models() -> dict[str, str]:
    """ Return a dict of available models and their corresponding Hugging Face model identifiers from the env file. .env file contains a variable MODELS which is a comma-separated list of model names and their corresponding Hugging Face model identifiers."""
    load_dotenv()
    models = os.getenv("MODELS")
    if not models:
        raise ValueError("Critical Error: MODELS not found.")
    return {model.split(":")[0].strip(): model.split(":")[1].strip() for model in models.split(",") if ":" in model}

if __name__ == "__main__":
    print(get_available_models())