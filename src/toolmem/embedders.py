"""Embedding providers. All return L2-normalized float32 arrays of shape (n, dim)."""
from __future__ import annotations

import re
import zlib
from typing import Protocol, Sequence, runtime_checkable

import numpy as np


def _normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    if m.ndim == 1:
        m = m[None, :]
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


@runtime_checkable
class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Dependency-free lexical embedder (word + char n-gram hashing).

    Not semantic. Use it for tests, CI, and offline smoke runs, or as a baseline in evals.
    """

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _features(self, text: str) -> list[str]:
        words = re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", text).lower())
        feats = [f"w:{w}" for w in words]
        feats += [f"b:{a}_{b}" for a, b in zip(words, words[1:])]
        for w in words:
            padded = f"#{w}#"
            feats += [f"c:{padded[i:i + 3]}" for i in range(len(padded) - 2)]
        return feats

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for f in self._features(t):
                h = zlib.crc32(f.encode())
                out[i, h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        return _normalize(out)


class SentenceTransformerEmbedder:
    """Local embeddings via sentence-transformers (pip install -e ".[local]")."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", **kwargs):
        from sentence_transformers import SentenceTransformer  # lazy import
        self._model = SentenceTransformer(model_name, **kwargs)
        self.name = f"st:{model_name}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return _normalize(self._model.encode(list(texts), convert_to_numpy=True))


class OpenAIEmbedder:
    """Any OpenAI-compatible /embeddings endpoint (pip install -e ".[openai]")."""

    def __init__(self, model: str = "text-embedding-3-small", client=None,
                 base_url: str | None = None, api_key: str | None = None, batch_size: int = 128):
        if client is None:
            from openai import OpenAI  # lazy import
            client = OpenAI(base_url=base_url, api_key=api_key)
        self._client, self.model, self.batch_size = client, model, batch_size
        self.name = f"openai:{model}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vecs: list[list[float]] = []
        texts = list(texts)
        for i in range(0, len(texts), self.batch_size):
            resp = self._client.embeddings.create(model=self.model, input=texts[i:i + self.batch_size])
            vecs.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        return _normalize(np.array(vecs))


class CallableEmbedder:
    """Wrap any function ``texts -> array-like`` as an embedder."""

    def __init__(self, fn, name: str = "callable"):
        self._fn, self.name = fn, name

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return _normalize(np.asarray(self._fn(list(texts))))
