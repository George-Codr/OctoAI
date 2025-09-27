# client.py
import asyncio, httpx, time, os, logging
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from aiolimiter import AsyncLimiter
from typing import Any, Dict
from config import KEYS, REQUEST_TIMEOUT, RETRY_MAX

logger = logging.getLogger("superbrain.client")

# conservative per-provider limits; tune to your quotas
LIMITERS = {
    "deepseek": AsyncLimiter(20, 1),
    "gemini": AsyncLimiter(10, 1),
    "groq": AsyncLimiter(15, 1),
    "openrouter": AsyncLimiter(15, 1),
}

BASES = {
    "deepseek": "https://api.deepseek.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://api.openrouter.ai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
}

class ProviderCircuit:
    def __init__(self):
        self.failures = {}
        self.open_until = {}

    def record_failure(self, name):
        self.failures[name] = self.failures.get(name, 0) + 1
        if self.failures[name] >= 3:
            self.open_until[name] = time.time() + 10

    def record_success(self, name):
        self.failures[name] = 0
        self.open_until[name] = 0

    def is_open(self, name):
        return time.time() < self.open_until.get(name, 0)

class UnifiedClient:
    def __init__(self):
        self._client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT, http2=True)
        self.circuit = ProviderCircuit()

    async def close(self):
        await self._client.aclose()

    @retry(wait=wait_exponential(multiplier=0.5, min=0.5, max=5), stop=stop_after_attempt(RETRY_MAX),
           retry=retry_if_exception_type(Exception))
    async def _post(self, provider: str, endpoint: str, payload: Dict[str, Any], headers: Dict[str, str]):
        if self.circuit.is_open(provider):
            raise RuntimeError(f"{provider} circuit open")
        limiter = LIMITERS.get(provider)
        base = BASES.get(provider)
        if base is None:
            raise ValueError("Unsupported provider")
        url = base + endpoint
        try:
            if limiter:
                async with limiter:
                    resp = await self._client.post(url, json=payload, headers=headers)
            else:
                resp = await self._client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            self.circuit.record_success(provider)
            return resp.json()
        except Exception as e:
            logger.warning("Provider %s error: %s", provider, e)
            self.circuit.record_failure(provider)
            raise

    # high-level chat wrapper
    async def chat(self, provider: str, model: str, messages, max_tokens=512, temperature=0.2):
        if provider == "deepseek":
            endpoint = "/chat/completions"
            headers = {"Authorization": f"Bearer {KEYS['deepseek']}", "Content-Type": "application/json"}
            payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
            return await self._post("deepseek", endpoint, payload, headers)

        if provider == "groq":
            endpoint = "/chat/completions"
            headers = {"Authorization": f"Bearer {KEYS['groq']}", "Content-Type": "application/json"}
            payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
            return await self._post("groq", endpoint, payload, headers)

        if provider == "openrouter":
            endpoint = "/chat/completions"
            headers = {"Authorization": f"Bearer {KEYS['openrouter']}", "Content-Type": "application/json"}
            payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
            return await self._post("openrouter", endpoint, payload, headers)

        if provider == "gemini":
            endpoint = f"/models/{model}:generateText"
            headers = {"Authorization": f"Bearer {KEYS['gemini']}", "Content-Type": "application/json"}
            payload = {"prompt": {"text": messages[-1]["content"]}, "maxOutputTokens": max_tokens, "temperature": temperature}
            return await self._post("gemini", endpoint, payload, headers)

        raise ValueError("Unknown provider")
