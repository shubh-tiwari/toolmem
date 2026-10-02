"""Model providers for the benchmark: one call = "given these tools, which one do you pick?"

Each provider implements ``select_tool(system, query, tools) -> Selection`` and an optional
``count_tokens`` used by ``run.py --dry-run``. Providers never see the expected answer.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Protocol, Sequence

from toolmem import Tool
from toolmem.bm25 import BM25Index

SYSTEM_PROMPT = (
    "You are an AI assistant that completes tasks by calling tools. Read the user's request and "
    "call the single tool that best accomplishes it. Do not ask clarifying questions; if a required "
    "argument is unknown, invent a plausible placeholder value. If no available tool fits the "
    "request, reply with the text NONE and do not call any tool."
)

# USD per million tokens: (input, output, cache_read, cache_write). Anthropic list prices as of
# 2026-09; OpenRouter models are looked up live by ``openrouter_price``.
PRICES: dict[str, tuple[float, float, float, float]] = {
    "claude-sonnet-5-5": (2.00, 10.00, 0.20, 2.50),
    "claude-sonnet-5": (2.00, 10.00, 0.20, 2.50),
    "claude-haiku-4-5": (1.00, 5.00, 0.10, 1.25),
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "fake": (0.0, 0.0, 0.0, 0.0),
    "jev-latest": (0.042, 0.0, 0.042, 0.042),  # TypeSafe list price: input only, output free
}


@dataclass
class Selection:
    tool: str | None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    latency_s: float = 0.0
    stop_reason: str = ""
    text: str = ""            # first 200 chars of any text reply (for debugging misses)
    error: str = ""           # non-empty when the call failed after retries
    reported_cost: float | None = None  # provider-reported USD, when available (OpenRouter)
    extra_cost: float = 0.0   # USD priced outside the provider's per-token table (mixed-model pipelines)
    spent: float | None = None  # USD the benchmark call actually cost, when it differs from the modeled cost

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "Selection":
        return cls(**json.loads(s))

    def cost(self, prices: tuple[float, float, float, float]) -> tuple[float, float]:
        """(list_price_usd, effective_usd). List price bills every input token at full rate;
        effective applies cache read/write rates."""
        pin, pout, pread, pwrite = prices
        total_in = self.input_tokens + self.cache_read + self.cache_write
        list_price = (total_in * pin + self.output_tokens * pout) / 1e6
        effective = (self.input_tokens * pin + self.cache_read * pread + self.cache_write * pwrite
                     + self.output_tokens * pout) / 1e6
        return list_price + self.extra_cost, effective + self.extra_cost


class Provider(Protocol):
    name: str
    model: str
    prices: tuple[float, float, float, float]

    def select_tool(self, system: str, query: str, tools: Sequence[Tool], cache: bool = False) -> Selection: ...

    def count_tokens(self, system: str, query: str, tools: Sequence[Tool]) -> int | None: ...


def _timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


# --------------------------------------------------------------------------- Anthropic
class AnthropicProvider:
    """Claude via the official ``anthropic`` SDK.

    ``tool_choice`` is ``auto`` (forced tool use is rejected by Sonnet 5.5), with parallel tool
    use disabled so at most one tool call comes back. With ``cache=True`` a ``cache_control``
    marker goes on the last tool so a tool list repeated across requests (the all-tools condition)
    is billed at cache-read rates; both raw and cached token counts are recorded.
    """

    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-5-5", effort: str = "low", max_tokens: int = 1024):
        import anthropic
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(max_retries=4)
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        self.prices = PRICES.get(model, PRICES["claude-sonnet-5-5"])

    def _request(self, system: str, query: str, tools: Sequence[Tool], cache: bool = False) -> dict:
        tool_defs = [t.to_anthropic() for t in tools]
        if cache and tool_defs:
            tool_defs[-1] = {**tool_defs[-1], "cache_control": {"type": "ephemeral"}}
        return dict(model=self.model, system=system, tools=tool_defs,
                    messages=[{"role": "user", "content": query}])

    def select_tool(self, system: str, query: str, tools: Sequence[Tool], cache: bool = False) -> Selection:
        """``cache=True`` only for tool lists repeated across requests: a cache write costs 1.25x."""
        req = self._request(system, query, tools, cache)
        try:
            resp, dt = _timed(lambda: self._client.messages.create(
                max_tokens=self.max_tokens,
                tool_choice={"type": "auto", "disable_parallel_tool_use": True},
                output_config={"effort": self.effort}, **req))
        except self._anthropic.APIError as e:  # after SDK retries
            return Selection(tool=None, error=f"{type(e).__name__}: {e}"[:300])
        tool_name, text = None, ""
        for block in resp.content:
            if block.type == "tool_use" and tool_name is None:
                tool_name = block.name
            elif block.type == "text" and not text:
                text = block.text[:200]
        u = resp.usage
        return Selection(tool=tool_name, input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                         cache_read=u.cache_read_input_tokens or 0,
                         cache_write=u.cache_creation_input_tokens or 0,
                         latency_s=dt, stop_reason=resp.stop_reason or "", text=text)

    def count_tokens(self, system: str, query: str, tools: Sequence[Tool]) -> int | None:
        req = self._request(system, query, tools)
        for d in req["tools"]:
            d.pop("cache_control", None)
        return self._client.messages.count_tokens(**req).input_tokens


# --------------------------------------------------------------------------- OpenRouter
def openrouter_price(model: str) -> tuple[float, float, float, float] | None:
    """Look up USD per million tokens from OpenRouter's public model listing."""
    try:
        with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=20) as r:
            data = json.load(r)["data"]
    except Exception:
        return None
    for m in data:
        if m.get("id") == model:
            p = m.get("pricing", {})
            pin, pout = float(p.get("prompt", 0)) * 1e6, float(p.get("completion", 0)) * 1e6
            pread = float(p.get("input_cache_read", 0) or 0) * 1e6 or pin
            pwrite = float(p.get("input_cache_write", 0) or 0) * 1e6 or pin
            return (pin, pout, pread, pwrite)
    return None


class OpenRouterProvider:
    """Any OpenAI-compatible chat model (default: OpenRouter). Used for DeepSeek and friends."""

    name = "openrouter"

    def __init__(self, model: str, base_url: str = "https://openrouter.ai/api/v1",
                 api_key: str | None = None, max_tokens: int = 1024):
        import openai
        self._openai = openai
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
                                     max_retries=4)
        self.model, self.max_tokens = model, max_tokens
        self.prices = PRICES.get(model) or openrouter_price(model) or (0.0, 0.0, 0.0, 0.0)
        self._parallel_arg_ok = True

    def select_tool(self, system: str, query: str, tools: Sequence[Tool], cache: bool = False) -> Selection:
        kwargs = dict(model=self.model, max_tokens=self.max_tokens, tool_choice="auto",
                      tools=[t.to_openai() for t in tools],
                      messages=[{"role": "system", "content": system}, {"role": "user", "content": query}],
                      extra_body={"usage": {"include": True}})
        if self._parallel_arg_ok:
            kwargs["parallel_tool_calls"] = False
        try:
            try:
                resp, dt = _timed(lambda: self._client.chat.completions.create(**kwargs))
            except self._openai.BadRequestError:
                if not self._parallel_arg_ok:
                    raise
                self._parallel_arg_ok = False  # some upstreams reject parallel_tool_calls
                kwargs.pop("parallel_tool_calls", None)
                resp, dt = _timed(lambda: self._client.chat.completions.create(**kwargs))
        except self._openai.APIError as e:
            return Selection(tool=None, error=f"{type(e).__name__}: {e}"[:300])
        if not resp.choices:
            return Selection(tool=None, error="empty choices")
        msg = resp.choices[0].message
        calls = msg.tool_calls or []
        tool_name = calls[0].function.name if calls else None
        u = resp.usage
        cached = 0
        details = getattr(u, "prompt_tokens_details", None) if u else None
        if details is not None:
            cached = getattr(details, "cached_tokens", 0) or 0
        cost = None
        if u is not None:
            extra = getattr(u, "model_extra", None) or {}
            cost = extra.get("cost", getattr(u, "cost", None))
        return Selection(tool=tool_name, input_tokens=(u.prompt_tokens - cached) if u else 0,
                         output_tokens=u.completion_tokens if u else 0, cache_read=cached,
                         latency_s=dt, stop_reason=resp.choices[0].finish_reason or "",
                         text=(msg.content or "")[:200], reported_cost=cost)

    def count_tokens(self, system: str, query: str, tools: Sequence[Tool]) -> int | None:
        return None  # run.py falls back to a character-based estimate


# --------------------------------------------------------------------------- Fake (offline)
class FakeProvider:
    """Offline stand-in: picks the tool whose text best matches the query by BM25.

    Deterministic and free. It is intentionally weak, so the offline test exercises the whole
    pipeline without asserting anything about real-model accuracy.
    """

    name = "fake"
    model = "fake"
    prices = PRICES["fake"]

    def select_tool(self, system: str, query: str, tools: Sequence[Tool], cache: bool = False) -> Selection:
        idx = BM25Index()
        for t in tools:
            idx.upsert(t.name, t.base_text())
        hits = idx.search(query, 1)
        n_in = self.count_tokens(system, query, tools) or 0
        return Selection(tool=hits[0][0] if hits else None, input_tokens=n_in, output_tokens=20,
                         latency_s=0.0, stop_reason="tool_use" if hits else "end_turn")

    def count_tokens(self, system: str, query: str, tools: Sequence[Tool]) -> int | None:
        chars = len(system) + len(query) + sum(len(json.dumps(t.to_openai())) for t in tools)
        return chars // 4


class JevProvider:
    """TypeSafe Jev as the picker: a Choice question over the shortlist instead of an LLM tool call.

    The system prompt is not sent; Jev gets its own instructions (see ``JevReranker``). Requests
    that fail are recorded as errors, not retried, so a missing credential shows up immediately.
    """

    name = "jev"

    def __init__(self, provider: str = "typesafe"):
        from toolmem import JevReranker
        self._r = JevReranker(provider=provider)
        self.model = self._r.model
        self.prices = PRICES["jev-latest"]

    def select_tool(self, system: str, query: str, tools: Sequence[Tool], cache: bool = False) -> Selection:
        try:
            d = self._r.rerank(query, tools)
        except Exception as e:  # noqa: BLE001
            return Selection(tool=None, error=f"{type(e).__name__}: {e}"[:300])
        return Selection(tool=d.name, input_tokens=int(d.usage.get("input_tokens", 0)),
                         output_tokens=int(d.usage.get("output_tokens", 0)), latency_s=d.latency_s,
                         stop_reason="choice", text=json.dumps({"confidence": d.confidence, "top3": d.top(3)}))

    def count_tokens(self, system: str, query: str, tools: Sequence[Tool]) -> int | None:
        return len(json.dumps(self._r.build_request(query, tools))) // 4


class JevRouteProvider:
    """Jev alone over a catalog too big for one question: pick the server, then the tool in it.

    Jev allows 255 options and about 32K tokens per question, so it cannot see 454 tools at once.
    Step 1 is a Choice over servers (each described by its tool names); step 2 is a Choice over
    that server's tools. Two requests per query; tokens are summed. Use with ``all@all``.
    """

    name = "jevroute"

    def __init__(self, provider: str = "typesafe"):
        from toolmem import JevReranker
        self._server = JevReranker(provider=provider, instructions=(
            "The state is a request a user sent to an AI assistant. Choose the service whose tools the "
            "assistant should use to carry out the request. Choose none only if no listed service can do it."))
        self._tool = JevReranker(provider=provider)
        self.model = "jev-latest-route"
        self.prices = PRICES["jev-latest"]

    @staticmethod
    def _servers(tools: Sequence[Tool]) -> dict[str, list[Tool]]:
        out: dict[str, list[Tool]] = {}
        for t in tools:
            out.setdefault(t.metadata.get("server") or t.name.split("__")[0], []).append(t)
        return out

    def _server_options(self, groups: dict[str, list[Tool]]) -> list[Tool]:
        opts = []
        for server, ts in groups.items():
            names = ", ".join(t.metadata.get("original_name", t.name) for t in ts)
            opts.append(Tool(name=server, description=f"{server} tools: {names}"))
        return opts

    def select_tool(self, system: str, query: str, tools: Sequence[Tool], cache: bool = False) -> Selection:
        groups = self._servers(tools)
        self._server.max_desc_chars = 3000
        try:
            d1 = self._server.rerank(query, self._server_options(groups))
            if d1.tool is None:
                return Selection(tool=None, input_tokens=int(d1.usage.get("input_tokens", 0)),
                                 latency_s=d1.latency_s, stop_reason="choice",
                                 text=json.dumps({"server": None, "server_confidence": d1.confidence}))
            d2 = self._tool.rerank(query, groups[d1.tool.name])
        except Exception as e:  # noqa: BLE001
            return Selection(tool=None, error=f"{type(e).__name__}: {e}"[:300])
        tok = int(d1.usage.get("input_tokens", 0)) + int(d2.usage.get("input_tokens", 0))
        return Selection(tool=d2.name, input_tokens=tok, latency_s=d1.latency_s + d2.latency_s, stop_reason="choice",
                         text=json.dumps({"server": d1.tool.name, "server_confidence": d1.confidence,
                                          "tool_confidence": d2.confidence}))

    def count_tokens(self, system: str, query: str, tools: Sequence[Tool]) -> int | None:
        groups = self._servers(tools)
        self._server.max_desc_chars = 3000
        first = len(json.dumps(self._server.build_request(query, self._server_options(groups)))) // 4
        biggest = max(groups.values(), key=len)
        return first + len(json.dumps(self._tool.build_request(query, biggest))) // 8  # about half the time a big server


class CascadeProvider:
    """Jev picks from the shortlist; below ``threshold`` confidence an LLM picks from Jev's top 3.

    For measurement, the LLM is asked on every query, so the result for any threshold can be
    computed afterwards from the recorded picks (``text`` holds both). The recorded ``tool``,
    cost and latency are what the cascade would do at ``threshold``: the LLM is counted only when
    Jev's confidence is below it. Cost is carried in ``extra_cost`` because it mixes two price
    tables.
    """

    name = "cascade"

    def __init__(self, llm_spec: str, threshold: float = 0.9, k: int = 3):
        self._jev = JevProvider()
        self._llm = make_provider(llm_spec)
        self.threshold, self.k = threshold, k
        self.model = f"cascade-jev-{self._llm.model.split('/')[-1]}"
        self.prices = (0.0, 0.0, 0.0, 0.0)

    def select_tool(self, system: str, query: str, tools: Sequence[Tool], cache: bool = False) -> Selection:
        try:
            d = self._jev._r.rerank(query, tools)
        except Exception as e:  # noqa: BLE001
            return Selection(tool=None, error=f"jev: {type(e).__name__}: {e}"[:300])
        by = {t.name: t for t in tools}
        top = [by[n] for n in d.top(self.k) if n in by] or list(tools[:self.k])
        llm = self._llm.select_tool(system, query, top)
        use_llm = (d.confidence or 0.0) < self.threshold or d.tool is None
        if llm.error and use_llm:  # a failed LLM call only matters when the cascade would use it
            return Selection(tool=None, error=f"llm: {llm.error}"[:300])
        jev_tok = int(d.usage.get("input_tokens", 0))
        jev_cost = jev_tok * PRICES["jev-latest"][0] / 1e6
        llm_cost = llm.reported_cost if llm.reported_cost is not None else llm.cost(self._llm.prices)[1]
        meta = {"jev_tool": d.name, "jev_confidence": d.confidence, "jev_top": [t.name for t in top],
                "llm_tool": llm.tool, "jev_tokens": jev_tok, "jev_latency_s": d.latency_s,
                "llm_input_tokens": llm.input_tokens + llm.cache_read, "llm_latency_s": llm.latency_s,
                "llm_cost": llm_cost, "jev_cost": jev_cost, "used_llm": use_llm, "llm_error": llm.error}
        return Selection(tool=llm.tool if use_llm else d.name,
                         input_tokens=jev_tok + (llm.input_tokens + llm.cache_read if use_llm else 0),
                         output_tokens=llm.output_tokens if use_llm else 0,
                         latency_s=d.latency_s + (llm.latency_s if use_llm else 0.0),
                         stop_reason="llm" if use_llm else "jev", text=json.dumps(meta),
                         extra_cost=jev_cost + (llm_cost if use_llm else 0.0),
                         spent=jev_cost + llm_cost)  # the LLM runs on every query, used or not

    def estimate_cost(self, system: str, query: str, tools: Sequence[Tool]) -> float:
        """Benchmark spend per query: one Jev call plus one LLM call on 3 tools (both always run)."""
        jev = (self._jev.count_tokens(system, query, tools) or 0) * PRICES["jev-latest"][0]
        llm_in = self._llm.count_tokens(system, query, tools[:self.k]) or (
            (len(system) + len(query) + sum(len(json.dumps(t.to_openai())) for t in tools[:self.k])) // 4)
        pin, pout = self._llm.prices[0], self._llm.prices[1]
        return (jev + llm_in * pin + 200 * pout) / 1e6

    def count_tokens(self, system: str, query: str, tools: Sequence[Tool]) -> int | None:
        return self._jev.count_tokens(system, query, tools)


def make_provider(spec: str, **kw) -> Provider:
    """``anthropic[:model]``, ``openrouter:<model>``, ``jev[:typesafe|cloudflare]`` or ``fake``."""
    kind, _, model = spec.partition(":")
    if kind == "anthropic":
        return AnthropicProvider(model or "claude-sonnet-5-5", **kw)
    if kind == "openrouter":
        if not model:
            raise ValueError("openrouter provider needs a model, e.g. openrouter:deepseek/deepseek-v4-pro")
        return OpenRouterProvider(model, **kw)
    if kind == "fake":
        return FakeProvider()
    if kind == "jev":
        return JevProvider(model or "typesafe")
    if kind == "jevroute":
        return JevRouteProvider(model or "typesafe")
    if kind == "cascade":  # cascade:<llm provider spec>, e.g. cascade:openrouter:deepseek/deepseek-v4-flash
        return CascadeProvider(model)
    raise ValueError(f"unknown provider {spec!r}")
