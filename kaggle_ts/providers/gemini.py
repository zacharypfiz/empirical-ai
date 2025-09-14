from __future__ import annotations

import os
from typing import List, Optional, Sequence

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from ..agents.base import LLMProvider


class _MissingDependency(Exception):
    pass

class GeminiAPIError(Exception):
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
        temperature: float = 1.0,
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
        # Raw env-provided thinking budget. Policy: only 0 (disable) or None.
        self._thinking_budget = thinking_budget

    async def generate(self, prompt: str, *, max_tokens: Optional[int] = None) -> str:
        # google-genai is sync; safe to run in thread if needed later.
        # Retry with slight config adjustments for robustness on empty/UNAVAILABLE.
        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                # Build config per attempt (adjust temperature / token budget on retries)
                temp = max(0.2, self._temperature * (0.7 if attempt > 0 else 1.0))
                cfg_kwargs = {"temperature": temp}
                m = max_tokens or self._max_output_tokens
                if m is not None:
                    # Reduce output target a bit on retries
                    cfg_kwargs["max_output_tokens"] = max(128, int(m * (0.8 if attempt > 0 else 1.0)))
                # Thinking policy: default is no config (thinking on by default).
                # If env is 0 and model supports disable, send budget=0 to disable.
                resolved_tb = _resolve_thinking_budget(
                    model=self._model, raw_budget=self._thinking_budget
                )
                if resolved_tb is not None:
                    cfg_kwargs["thinking_config"] = self._types.ThinkingConfig(
                        thinking_budget=resolved_tb
                    )
                cfg = self._types.GenerateContentConfig(**cfg_kwargs)

                resp = await _to_thread(
                    self._client.models.generate_content,
                    model=self._model,
                    contents=prompt,
                    config=cfg,
                )
                # Robust text extraction across SDK variants
                text = getattr(resp, "text", None) or getattr(resp, "output_text", None)
                if not text:
                    try:
                        parts = []
                        for cand in getattr(resp, "candidates", []) or []:
                            content = getattr(cand, "content", None)
                            for part in getattr(content, "parts", []) or []:
                                t = getattr(part, "text", None)
                                if t:
                                    parts.append(t)
                        if parts:
                            text = "\n".join(parts)
                    except Exception:
                        text = None
                if not text:
                    # Trigger retry with adjusted config
                    raise GeminiAPIError("Empty response text")
                return text
            except Exception as e:
                last_err = e
                msg = str(e)
                # Retry on empties and transient 503/UNAVAILABLE once or twice
                if attempt < 2 and ("503" in msg or "UNAVAILABLE" in msg or isinstance(e, GeminiAPIError)):
                    continue
                break
        raise GeminiAPIError(f"Gemini generate failed: {last_err}")


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


# --- Thinking helpers -------------------------------------------------------

def _model_supports_thinking_disable(model: str) -> bool:
    """True if model supports disabling thinking via budget=0 (Flash only)."""
    m = model.lower()
    return ("2.5" in m) and ("flash" in m)


def _resolve_thinking_budget(*, model: str, raw_budget: Optional[int]) -> Optional[int]:
    """Return 0 to disable thinking when supported; otherwise None.

    - Default: None (no config; thinking stays enabled by default)
    - If raw_budget == 0 and model is a 2.5 Flash variant: return 0
    - Any other value: None
    - Never send config for models like 2.5 Pro
    """
    if not _model_supports_thinking_disable(model):
        return None
    if raw_budget == 0:
        return 0
    return None
