import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# 1. Securely load environment variables
load_dotenv()

def ping_deepseek_with_client(prompt_text):
    """
    Sends a prompt to DeepSeek R1 using the official InferenceClient.
    """
    api_token = os.getenv("HF_API_TOKEN")
    if not api_token:
        raise ValueError("Critical Error: HF_API_TOKEN not found.")

    # Initialize the client (it automatically handles the headers and base URL)
    client = InferenceClient(
        token=api_token
    )
    
    print(f"Pinging DeepSeek R1 securely...\nPrompt: '{prompt_text}'\n")
    
    try:
        completion = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1:novita",
            messages=[
                {
                    "role": "user",
                    "content": prompt_text
                }
            ],
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        return f"API Request failed: {e}"

if __name__ == "__main__":
    result = ping_deepseek_with_client("Explain the concept of Forward Deployed Engineering in exactly two sentences.")
    print("--- Model Response ---")
    print(result)