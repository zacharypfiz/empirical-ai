from __future__ import annotations

import os
from typing import Optional
from google import genai
from google.genai import types

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from ..agents.base import LLMProvider


def _extract_text(resp) -> str:
    """Best-effort extraction of text from google-genai responses."""
    text = getattr(resp, "text", None) or getattr(resp, "output_text", None)
    if text:
        return text
    # Aggregate text from candidates.finish_message/content parts
    try:
        acc: list[str] = []
        for cand in getattr(resp, "candidates", []) or []:
            fm = getattr(cand, "finish_message", None)
            if fm is not None:
                content = getattr(fm, "content", None)
                if content is not None:
                    parts = getattr(content, "parts", None)
                    if parts:
                        for p in parts:
                            t = getattr(p, "text", None)
                            if t:
                                acc.append(t)
            content = getattr(cand, "content", None)
            if content is not None:
                parts = getattr(content, "parts", None)
                if parts:
                    for p in parts:
                        t = getattr(p, "text", None)
                        if t:
                            acc.append(t)
        if acc:
            return "\n".join(acc)
    except Exception:
        pass
    # Top-level parts fallback
    try:
        acc2: list[str] = []
        for part in getattr(resp, "parts", []) or []:
            t = getattr(part, "text", None)
            if t:
                acc2.append(t)
        if acc2:
            return "\n".join(acc2)
    except Exception:
        pass
    # Try SDK private helper if present (last resort)
    getter = getattr(resp, "_get_text", None)
    if callable(getter):
        try:
            t = getter()
            if t:
                return t
        except Exception:
            pass
    return ""


class _BaseGemini(LLMProvider):
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str,
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None,
        thinking_budget: Optional[int] = None,
        system_instruction: Optional[str] = None,
    ) -> None:
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:  # pragma: no cover
            raise ValueError("GEMINI_API_KEY not set")

        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        if thinking_budget is None:
            budget_env = os.getenv("GEMINI_THINKING_BUDGET")
            if budget_env is not None and budget_env.strip():
                thinking_budget = int(budget_env)
        self._thinking_budget = thinking_budget
        self._system_instruction = system_instruction

    async def generate(self, prompt: str, *, max_tokens: Optional[int] = None) -> str:
        import asyncio

        cfg_kwargs = {"temperature": self._temperature}
        m = max_tokens or self._max_output_tokens
        if m is not None:
            cfg_kwargs["max_output_tokens"] = m
        if self._system_instruction is not None:
            cfg_kwargs["system_instruction"] = self._system_instruction
        if self._thinking_budget is not None:
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self._thinking_budget
            )

        config = types.GenerateContentConfig(**cfg_kwargs)
        resp = await asyncio.to_thread(
            self._client.models.generate_content,
            model=self._model,
            contents=prompt,
            config=config,
        )
        return _extract_text(resp)


class GeminiFlashProvider(_BaseGemini):
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        temperature: float = 1.0,
        max_output_tokens: Optional[int] = None,
        thinking_budget: Optional[int] = None,
        system_instruction: Optional[str] = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model="gemini-2.5-flash",
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            thinking_budget=thinking_budget,
            system_instruction=system_instruction,
        )


class GeminiProProvider(_BaseGemini):
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None,
        system_instruction: Optional[str] = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model="gemini-2.5-pro",
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system_instruction=system_instruction,
        )


class GeminiProvider(LLMProvider):
    """Compatibility shim. Prefer GeminiFlashProvider or GeminiProProvider."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-pro",
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None,
        thinking_budget: Optional[int] = None,
        system_instruction: Optional[str] = None,
    ) -> None:
        model_l = (model or "").lower()
        if "flash" in model_l:
            self._inner: LLMProvider = GeminiFlashProvider(
                api_key=api_key,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                thinking_budget=thinking_budget,
                system_instruction=system_instruction,
            )
        else:
            self._inner = GeminiProProvider(
                api_key=api_key,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                system_instruction=system_instruction,
            )
        # Expose model name for simple introspection in tests
        self._model = getattr(self._inner, "_model", None)

    async def generate(self, prompt: str, *, max_tokens: Optional[int] = None) -> str:
        return await self._inner.generate(prompt, max_tokens=max_tokens)


class GeminiEmbeddings:
    def __init__(self, *, api_key: Optional[str] = None, model: str = "gemini-embedding-001") -> None:
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:  # pragma: no cover
            raise ValueError("GEMINI_API_KEY not set")
        self._client = genai.Client(api_key=api_key)
        self._model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import asyncio

        contents = list(texts)
        config = types.EmbedContentConfig(task_type="CLUSTERING")
        resp = await asyncio.to_thread(
            self._client.models.embed_content,
            model=self._model,
            contents=contents,
            config=config,
        )
        if hasattr(resp, "embeddings") and isinstance(resp.embeddings, list):
            return [getattr(e, "values", getattr(e, "embedding", [])) for e in resp.embeddings]
        emb = getattr(resp, "embedding", None) or getattr(resp, "values", None)
        if emb is None:
            return []
        return [emb]
