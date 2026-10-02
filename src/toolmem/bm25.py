"""Small dependency-free BM25 index for exact-name and keyword matching."""
from __future__ import annotations

import math
import re
from collections import Counter

_STOP = frozenset("a an the of to for in on and or is are be with by from at as it this that "
                  "what how do does can i me my you your please".split())


def tokenize(text: str) -> list[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._docs: dict[str, Counter] = {}
        self._lens: dict[str, int] = {}
        self._df: Counter = Counter()

    def upsert(self, id: str, text: str) -> None:
        self.delete(id)
        toks = tokenize(text)
        tf = Counter(toks)
        self._docs[id], self._lens[id] = tf, len(toks)
        self._df.update(tf.keys())

    def delete(self, id: str) -> None:
        tf = self._docs.pop(id, None)
        if tf is not None:
            self._lens.pop(id)
            self._df.subtract(tf.keys())

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        q = set(tokenize(query))
        n = len(self._docs)
        if not q or not n:
            return []
        avg = sum(self._lens.values()) / n or 1.0
        scores: dict[str, float] = {}
        for id, tf in self._docs.items():
            s = 0.0
            for t in q:
                f = tf.get(t)
                if not f:
                    continue
                idf = math.log(1 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5))
                s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self._lens[id] / avg))
            if s > 0:
                scores[id] = s
        return sorted(scores.items(), key=lambda x: -x[1])[:k]
