"""LLM augmentation: rewrite tool descriptions into retrieval-friendly text.

The augmented text is used only for embedding and keyword search. The tool schema sent
to the agent keeps the original description.
"""
from __future__ import annotations

import hashlib
import warnings
from typing import Callable, Protocol

from .cache import SQLiteCache
from .tool import Tool

DEFAULT_PROMPT = """You write search-index text for an AI agent's tool catalog.
Given a tool definition, write a plain-text description (no markdown, under 120 words) that:
1. States what the tool does and what it returns, in concrete terms.
2. Says when to use it and when NOT to (name the nearest confusable task).
3. Lists 4-6 short example user requests that should trigger it, phrased the way real users talk, using varied vocabulary and synonyms.
Do not invent capabilities the definition does not imply.

Tool definition:
{tool}"""


class Augmenter(Protocol):
    name: str

    def augment(self, tool: Tool) -> str: ...


class LLMAugmenter:
    """Augment with any OpenAI-compatible chat model (OpenAI, OpenRouter, vLLM, Ollama...).

    Example (OpenRouter)::

        LLMAugmenter(model="deepseek/deepseek-v4.1-flash",
                     base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"],
                     extra_body={"reasoning": {"enabled": False}})

    Reasoning models can spend the whole ``max_tokens`` budget on hidden reasoning and return no
    text. Turn reasoning off through ``extra_body`` (the OpenRouter form is shown above) or raise
    ``max_tokens``; an empty result is not cached and triggers a warning.
    """

    def __init__(self, model: str, client=None, base_url: str | None = None, api_key: str | None = None,
                 prompt: str = DEFAULT_PROMPT, cache: SQLiteCache | None = None,
                 temperature: float = 0.2, max_tokens: int = 300, extra_body: dict | None = None):
        if client is None:
            from openai import OpenAI  # lazy import
            client = OpenAI(base_url=base_url, api_key=api_key)
        self._client, self.model, self.prompt = client, model, prompt
        self.cache, self.temperature, self.max_tokens = cache, temperature, max_tokens
        self.extra_body = extra_body
        self._warned = False
        self.name = f"llm:{model}:{hashlib.sha256(prompt.encode()).hexdigest()[:8]}"

    def augment(self, tool: Tool) -> str:
        key = ("augment", self.name, tool.content_hash())
        if self.cache and (hit := self.cache.get_text(*key)) is not None:
            return hit
        kwargs = {"extra_body": self.extra_body} if self.extra_body else {}
        resp = self._client.chat.completions.create(
            model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": self.prompt.format(tool=tool.base_text())}], **kwargs)
        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        if not text and not self._warned:
            self._warned = True
            hint = (" The output limit was reached, so a reasoning model probably used it all; turn "
                    "reasoning off via extra_body or raise max_tokens.") if choice.finish_reason == "length" else ""
            warnings.warn(f"LLMAugmenter got an empty description from {self.model} for '{tool.name}'.{hint}",
                          RuntimeWarning, stacklevel=2)
        if self.cache and text:
            self.cache.set_text(text, *key)
        return text


class CallableAugmenter:
    """Wrap any ``Tool -> str`` function (handy for tests or custom pipelines)."""

    def __init__(self, fn: Callable[[Tool], str], name: str = "callable"):
        self._fn, self.name = fn, name

    def augment(self, tool: Tool) -> str:
        return self._fn(tool)
