# AI Sandbox

A Streamlit chat UI backed by the Hugging Face Inference API, with per-conversation
model selection and reasoning-model "thinking" traces.

## Setup

1. `pip install -r requirements.txt`
2. Create a `.env` file in the repo root:
   ```
   HF_API_TOKEN="hf_..."
   MODELS="deepseekR1: deepseek-ai/DeepSeek-R1:novita, Qwen3: qwen/Qwen-3.0-Chat"
   ```
3. `streamlit run app.py`

Conversations live in the browser session only (`st.session_state`) -- restarting
the app clears history.
