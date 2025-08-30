import os

# Hugging Face token (needed for Granite + other HF models)
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Granite model ID (IBM’s LLMs on Hugging Face)
GRANITE_MODEL_ID = os.getenv("GRANITE_MODEL_ID", "ibm-granite/granite-13b-chat-v2")

# API server config
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_URL = f"http://{API_HOST}:{API_PORT}"
