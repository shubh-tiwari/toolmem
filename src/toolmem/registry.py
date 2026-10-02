"""ToolRegistry: store tools, index them, and retrieve the right ones per query."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, Sequence

import numpy as np

from .augment import Augmenter
from .backends import InMemoryBackend, VectorBackend
from .bm25 import BM25Index
from .cache import SQLiteCache
from .embedders import Embedder, HashingEmbedder
from .rerank import Decision, Reranker
from .tool import Tool

Mode = Literal["hybrid", "semantic", "keyword"]
Format = Literal["openai", "anthropic", "mcp"]


@dataclass
class SearchResult:
    tool: Tool
    score: float
    semantic_rank: int | None = None
    keyword_rank: int | None = None

    @property
    def name(self) -> str:
        return self.tool.name


class ToolRegistry:
    """Semantic tool memory.

    Args:
        embedder: Embedding provider. Defaults to the dependency-free ``HashingEmbedder``.
        backend: Vector index. Defaults to exact in-memory search.
        augmenter: Optional LLM augmenter applied to each tool at index time.
        cache: Optional ``SQLiteCache`` for embeddings (augmenters take their own cache).
        semantic_weight: Weight of semantic vs keyword ranks in hybrid mode (0..1).
        rrf_k: Reciprocal-rank-fusion constant.
        max_workers: Parallel augmentation calls.
        reranker: Optional ``Reranker`` used by ``select`` to pick one tool from a shortlist.
    """

    def __init__(self, embedder: Embedder | None = None, backend: VectorBackend | None = None,
                 augmenter: Augmenter | None = None, cache: SQLiteCache | None = None,
                 semantic_weight: float = 0.6, rrf_k: int = 60, max_workers: int = 8,
                 reranker: Reranker | None = None):
        self.embedder = embedder or HashingEmbedder()
        self.backend = backend if backend is not None else InMemoryBackend()  # an empty backend is falsy
        self.augmenter, self.cache, self.reranker = augmenter, cache, reranker
        self.semantic_weight, self.rrf_k, self.max_workers = semantic_weight, rrf_k, max_workers
        self._tools: dict[str, Tool] = {}
        self._bm25 = BM25Index()
        self._dirty: set[str] = set()

    # ---------- registration ----------
    def register(self, func: Callable | None = None, *, name: str | None = None,
                 description: str | None = None, tags: list[str] | None = None):
        """Decorator: ``@registry.register`` or ``@registry.register(tags=[...])``."""
        def deco(f: Callable) -> Callable:
            self.add(Tool.from_function(f, name=name, description=description, tags=tags))
            return f
        return deco(func) if func is not None else deco

    def add(self, tool: Tool) -> Tool:
        self._tools[tool.name] = tool
        self._dirty.add(tool.name)
        return tool

    def add_many(self, tools: Iterable[Tool]) -> None:
        for t in tools:
            self.add(t)

    def remove(self, name: str) -> None:
        self._tools.pop(name, None)
        self._dirty.discard(name)
        self.backend.delete([name])
        self._bm25.delete(name)

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())

    # ---------- indexing ----------
    def index(self) -> None:
        """Augment and embed any new or changed tools. Called automatically by ``search``."""
        if not self._dirty:
            return
        pending = [self._tools[n] for n in sorted(self._dirty)]
        if self.augmenter:
            todo = [t for t in pending if not t.augmented_description]
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                for t, text in zip(todo, ex.map(self.augmenter.augment, todo)):
                    t.augmented_description = text or None
        texts = [t.embedding_text() for t in pending]
        vecs = self._embed(texts)
        self.backend.upsert([t.name for t in pending], vecs)
        for t, text in zip(pending, texts):
            self._bm25.upsert(t.name, text)
        self._dirty.clear()

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        if not self.cache:
            return self.embedder.embed(texts)
        out: list[np.ndarray | None] = [self.cache.get_embedding(self.embedder.name, t) for t in texts]
        missing = [i for i, v in enumerate(out) if v is None]
        if missing:
            fresh = self.embedder.embed([texts[i] for i in missing])
            for i, v in zip(missing, fresh):
                self.cache.set_embedding(self.embedder.name, texts[i], v)
                out[i] = v
        return np.stack(out)  # type: ignore[arg-type]

    # ---------- retrieval ----------
    def search(self, query: str, k: int = 5, mode: Mode = "hybrid",
               tags: Sequence[str] | None = None) -> list[SearchResult]:
        self.index()
        if not self._tools:
            return []
        pool = len(self._tools) if tags else max(k * 5, 25)
        sem = self.backend.search(self._embed([query])[0], pool) if mode != "keyword" else []
        kw = self._bm25.search(query, pool) if mode != "semantic" else []

        def allowed(name: str) -> bool:
            return name in self._tools and (not tags or bool(set(tags) & set(self._tools[name].tags)))

        sem = [(n, s) for n, s in sem if allowed(n)]
        kw = [(n, s) for n, s in kw if allowed(n)]
        if mode == "semantic":
            return [SearchResult(self._tools[n], s, semantic_rank=i + 1) for i, (n, s) in enumerate(sem[:k])]
        if mode == "keyword":
            return [SearchResult(self._tools[n], s, keyword_rank=i + 1) for i, (n, s) in enumerate(kw[:k])]

        w = self.semantic_weight
        sem_rank = {n: i + 1 for i, (n, _) in enumerate(sem)}
        kw_rank = {n: i + 1 for i, (n, _) in enumerate(kw)}
        fused = {}
        for n in set(sem_rank) | set(kw_rank):
            s = 0.0
            if n in sem_rank:
                s += w / (self.rrf_k + sem_rank[n])
            if n in kw_rank:
                s += (1 - w) / (self.rrf_k + kw_rank[n])
            fused[n] = s
        ranked = sorted(fused.items(), key=lambda x: -x[1])[:k]
        return [SearchResult(self._tools[n], s, sem_rank.get(n), kw_rank.get(n)) for n, s in ranked]

    def select(self, query: str, candidates: int = 40, mode: Mode = "hybrid",
               tags: Sequence[str] | None = None) -> Decision:
        """Retrieve a shortlist of ``candidates`` tools, then let the reranker pick one.

        Without a reranker this returns the top retrieved tool with ``confidence=None``.
        """
        shortlist = [r.tool for r in self.search(query, k=candidates, mode=mode, tags=tags)]
        if self.reranker is None:
            return Decision(tool=shortlist[0] if shortlist else None, confidence=None,
                            ranked=[(t.name, 0.0) for t in shortlist])
        if not shortlist:
            return Decision(tool=None, confidence=None)
        return self.reranker.rerank(query, shortlist)

    def tools_for(self, query: str, k: int = 5, format: Format = "openai", **kw) -> list[dict]:
        """Retrieve and export tool schemas ready to pass to an LLM API."""
        return [self._export(r.tool, format) for r in self.search(query, k, **kw)]

    @staticmethod
    def _export(tool: Tool, format: Format) -> dict:
        return {"openai": tool.to_openai, "anthropic": tool.to_anthropic, "mcp": tool.to_mcp}[format]()

    # ---------- execution ----------
    def call(self, name: str, arguments: dict | str | None = None) -> Any:
        """Execute a tool by name. ``arguments`` may be a dict or a JSON string."""
        if isinstance(arguments, str):
            arguments = json.loads(arguments or "{}")
        return self._tools[name](**(arguments or {}))

    # ---------- toolbox meta-tool ----------
    SEARCH_TOOL_NAME = "search_tools"

    def search_tool(self, format: Format = "openai") -> dict:
        """A meta-tool the agent can call to discover tools on demand."""
        meta = Tool(name=self.SEARCH_TOOL_NAME,
                    description="Search the tool catalog for tools that can help with a task. "
                                "Call this when none of your current tools fit. Returns tool definitions "
                                "you can call afterwards.",
                    parameters={"type": "object", "properties": {
                        "query": {"type": "string", "description": "Plain-language description of the task."},
                        "k": {"type": "integer", "description": "How many tools to return.", "default": 5}},
                        "required": ["query"]})
        return self._export(meta, format)

    def handle_search_tool(self, arguments: dict | str, format: Format = "openai") -> list[dict]:
        if isinstance(arguments, str):
            arguments = json.loads(arguments or "{}")
        return self.tools_for(arguments["query"], int(arguments.get("k", 5)), format=format)

    # ---------- persistence (definitions only; functions are re-attached by name) ----------
    def save(self, path: str) -> None:
        data = [{"name": t.name, "description": t.description, "parameters": t.parameters,
                 "tags": t.tags, "augmented_description": t.augmented_description,
                 "metadata": t.metadata} for t in self._tools.values()]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str, functions: dict[str, Callable] | None = None) -> None:
        functions = functions or {}
        with open(path, encoding="utf-8") as f:
            for d in json.load(f):
                self.add(Tool(func=functions.get(d["name"]), **d))
