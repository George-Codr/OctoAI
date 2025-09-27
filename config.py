# config.py
import os
from dotenv import load_dotenv
load_dotenv()

# --------------------------
# API keys
# --------------------------
API_KEYS = {
    "deepseek": os.getenv("DEEPSEEK_API_KEY"),
    "gemini": os.getenv("GEMINI_API_KEY"),
    "openrouter": os.getenv("OPENROUTER_API_KEY")
}

# --------------------------
# Models per capability
# --------------------------
MODELS = {
    "reasoning": [("deepseek", "DeepSeek-V3")],
    "search": [
        ("gemini", "gemini-1.5-pro"),
        ("gemini", "gemini-1.5-flash"),
        ("openrouter", "perplexity-latest")
    ],
    "coding": [
        ("deepseek", "DeepSeek-Coder"),
        ("deepseek", "DeepSeek-R1")
    ],
    "creativity": [
        ("deepseek", "DeepSeek-Creative")
    ],
    "summarizer": [
        ("gemini", "gemini-1.5-pro"),
        ("openrouter", "perplexity-latest")
    ],
    "sensory": [
        ("gemini", "gemini-1.5-flash")
    ]
}

# --------------------------
# Provider baseline trust (learned later)
# --------------------------
PROVIDER_PRIORS = {
    "deepseek": 1.0,
    "gemini": 0.95,
    "openrouter": 0.9
}
# --------------------------
# Runtime & resilience
# --------------------------
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "20"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
RETRY_MAX = int(os.getenv("RETRY_MAX", "3"))

# Storage paths
DATA_DIR = os.getenv("DATA_DIR", "./.sb_data")
VECTOR_DB_PATH = os.path.join(DATA_DIR, "vectors.faiss")
SQL_DB_PATH = os.path.join(DATA_DIR, "memory.sqlite")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
