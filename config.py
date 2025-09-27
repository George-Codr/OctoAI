# config.py
import os
from dotenv import load_dotenv
load_dotenv()

# --------------------------
# API keys
# --------------------------
KEYS = {
    "deepseek": os.getenv("DEEPSEEK_API_KEY"),
    "gemini": os.getenv("GEMINI_API_KEY"),
    "groq": os.getenv("GROQ_API_KEY"),
    "openrouter": os.getenv("OPENROUTER_API_KEY"),
    # Optional web verifier (Google Custom Search / SerpAPI / Bing)
    "web_search": os.getenv("WEB_SEARCH_API_KEY"),
}

# --------------------------
# Models per capability
# --------------------------
MODELS = {
    "sensory": ("groq", "llama-3.1-8b-instant"),   # fast classifier
    "reasoning": [("deepseek", "DeepSeek-R1"), ("groq", "llama-3.1-70b")],
    "search": [("gemini", "gemini-1.5-pro"), ("openrouter", "perplexity-latest")],
    "coding": [("deepseek", "DeepSeek-Coder"), ("groq", "mixtral-8x7b")],
    "creativity": [("openrouter", "gpt-4o-mini"), ("openrouter", "claude-3.5-sonnet")],
    "summarizer": [("deepseek", "DeepSeek-V3"), ("openrouter", "gpt-4o-mini")],
}

# --------------------------
# Provider baseline trust (learned later)
# --------------------------
PROVIDER_PRIORS = {"deepseek": 0.95, "gemini": 0.9, "groq": 0.85, "openrouter": 0.88}

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
