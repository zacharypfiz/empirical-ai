from __future__ import annotations

import os
from typing import List, Optional, Sequence

from ..agents.base import LLMProvider


class _MissingDependency(Exception):
    pass


def _load_genai():
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
    except Exception as e:  # pragma: no cover
        raise _MissingDependency(
            "google-genai is not installed. Install via: uv add google-genai"
        ) from e
    return genai, types


class GeminiProvider(LLMProvider):
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.6,
        max_output_tokens: Optional[int] = None,
        thinking_budget: Optional[int] = None,
    ) -> None:
        genai, types = _load_genai()
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:  # pragma: no cover
            raise ValueError("GEMINI_API_KEY not set")
        self._genai = genai
        self._types = types
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._thinking_budget = thinking_budget

    async def generate(self, prompt: str, *, max_tokens: Optional[int] = None) -> str:
        cfg_kwargs = {"temperature": self._temperature}
        m = max_tokens or self._max_output_tokens
        if m is not None:
            cfg_kwargs["max_output_tokens"] = m
        if self._thinking_budget is not None:
            cfg_kwargs["thinking_config"] = self._types.ThinkingConfig(
                thinking_budget=self._thinking_budget
            )
        cfg = self._types.GenerateContentConfig(**cfg_kwargs)
        # google-genai is sync; safe to run in thread if needed later.
        resp = await _to_thread(self._client.models.generate_content, self._model, prompt, cfg)
        # Prefer .text; fall back to string representation
        return getattr(resp, "text", str(resp))


class GeminiEmbeddings:
    def __init__(self, *, api_key: Optional[str] = None, model: str = "gemini-embedding-001") -> None:
        genai, _ = _load_genai()
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:  # pragma: no cover
            raise ValueError("GEMINI_API_KEY not set")
        self._client = genai.Client(api_key=api_key)
        self._model = model

    async def embed(self, texts: Sequence[str]) -> List[List[float]]:
        # The embed API supports single string or list of strings.
        contents: Sequence[str] = list(texts)
        resp = await _to_thread(self._client.models.embed_content, self._model, contents)
        # Normalize result to list of embeddings
        if hasattr(resp, "embeddings") and isinstance(resp.embeddings, list):
            # Multiple inputs
            return [getattr(e, "values", getattr(e, "embedding", [])) for e in resp.embeddings]
        # Single input
        emb = getattr(resp, "embedding", None) or getattr(resp, "values", None)
        if emb is None:  # pragma: no cover
            return []
        return [emb]


async def _to_thread(func, *args, **kwargs):
    import asyncio

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

