"""Rerankers: pick one tool from a retrieved shortlist, with a confidence.

Retrieval narrows thousands of tools to a shortlist of 20 to 50 cheaply. A reranker then chooses
among the shortlist. ``ToolRegistry.select`` wires the two together and returns a ``Decision``.

The interface is vendor-neutral. ``JevReranker`` is the first implementation: TypeSafe's Jev model
answers a Choice question over the candidates and returns a probability for each option.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, Sequence

if TYPE_CHECKING:
    from .tool import Tool

NO_TOOL = "none"


@dataclass
class Decision:
    """The outcome of ``ToolRegistry.select``.

    ``tool`` is ``None`` when the reranker judged that no candidate fits. ``confidence`` is the
    reranker's probability for its pick, or ``None`` when it gives none (retrieval-only fallback).
    ``ranked`` lists every candidate with its probability, highest first.
    """

    tool: "Tool | None"
    confidence: float | None
    ranked: list[tuple[str, float]] = field(default_factory=list)
    needs_tool: float | None = None     # P(the request needs any tool), when asked
    latency_s: float = 0.0
    usage: dict = field(default_factory=dict)

    @property
    def name(self) -> str | None:
        return self.tool.name if self.tool else None

    def top(self, n: int = 3) -> list[str]:
        """The ``n`` most likely tool names, for handing a short list to an LLM when confidence is low."""
        return [name for name, _ in self.ranked if name != NO_TOOL][:n]


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, candidates: Sequence["Tool"]) -> Decision: ...


def _option_text(tool: "Tool", max_chars: int) -> str:
    params = ", ".join((tool.parameters.get("properties") or {}).keys())
    text = tool.description.strip().replace("\n", " ")
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "..."
    return f"{text} (arguments: {params})" if params else text


class JevReranker:
    """Choose among candidates with TypeSafe's Jev (a typed-decision model, not a text generator).

    Each candidate becomes a Choice option whose key is the tool name and whose criteria text is
    the tool description, so the model sees the same original descriptions an LLM would. A
    ``none`` option lets it say no candidate fits. With ``ask_needs_tool=True`` a second question
    (a Noul, i.e. a yes/no probability) asks whether the request needs a tool at all, in the same
    request.

    Two access routes use the same request body:

    * ``provider="typesafe"`` (default): ``POST https://api.typesafe.ai/v1/systemone`` with
      ``TYPESAFE_API_KEY``. Access is invite-only.
    * ``provider="cloudflare"``: Workers AI model ``typesafe/jev`` through Cloudflare's REST API
      with ``CLOUDFLARE_API_TOKEN`` and ``CLOUDFLARE_ACCOUNT_ID``. This route follows Cloudflare's
      standard ``/ai/run/{model}`` pattern and has not been verified against a live account yet.

    Jev allows up to 255 options and about 32K tokens per question plus input; ``max_desc_chars``
    keeps a 50-tool shortlist well inside that.
    """

    TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
    CLOUDFLARE_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/typesafe/jev"
    MAX_OPTIONS = 255

    def __init__(self, api_key: str | None = None, provider: str = "typesafe", model: str = "jev-latest",
                 account_id: str | None = None, url: str | None = None, ask_needs_tool: bool = False,
                 max_desc_chars: int = 400, timeout: float = 30.0, instructions: str | None = None):
        if provider not in ("typesafe", "cloudflare"):
            raise ValueError("provider must be 'typesafe' or 'cloudflare'")
        self.provider, self.model = provider, model
        if provider == "typesafe":
            self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
            self.url = url or self.TYPESAFE_URL
        else:
            self.api_key = api_key or os.environ.get("CLOUDFLARE_API_TOKEN")
            account_id = account_id or os.environ.get("CLOUDFLARE_ACCOUNT_ID")
            if not url and not account_id:
                raise ValueError("cloudflare provider needs account_id or CLOUDFLARE_ACCOUNT_ID")
            self.url = url or self.CLOUDFLARE_URL.format(account_id=account_id)
        if not self.api_key:
            raise ValueError(f"no API key for Jev provider {provider!r}")
        self.ask_needs_tool, self.max_desc_chars, self.timeout = ask_needs_tool, max_desc_chars, timeout
        self.instructions = instructions or (
            "The state is a request a user sent to an AI assistant. Choose the tool the assistant should "
            "call to carry out the request. Choose none only if no listed tool can do it.")
        self.name = f"jev:{provider}:{model}"

    # -- request building is separate from sending so it can be tested without network access
    def build_request(self, query: str, candidates: Sequence["Tool"]) -> dict:
        if len(candidates) >= self.MAX_OPTIONS:
            raise ValueError(f"Jev allows at most {self.MAX_OPTIONS} options including 'none'; got {len(candidates)} candidates")
        criteria = {t.name: _option_text(t, self.max_desc_chars) for t in candidates}
        criteria[NO_TOOL] = "No listed tool can carry out this request."
        questions: dict = {"tool": {"type": "choice", "instructions": self.instructions, "criteria": criteria}}
        if self.ask_needs_tool:
            questions["needs_tool"] = {
                "type": "noul",
                "instructions": "Does carrying out this request require calling an external tool or service, "
                                "rather than only replying in text?"}
        return {"model": self.model, "state": query, "questions": questions}

    def parse_response(self, data: dict, candidates: Sequence["Tool"]) -> Decision:
        if "result" in data and "answers" not in data:  # Cloudflare wraps model output in "result"
            data = data["result"]
        answer = data["answers"]["tool"]
        probs = {k: float(v) for k, v in (answer.get("probabilities") or {}).items()}
        pick = answer.get("choice")
        by_name = {t.name: t for t in candidates}
        tool = by_name.get(pick) if pick != NO_TOOL else None
        confidence = probs.get(pick, answer.get("confidence"))
        ranked = sorted(probs.items(), key=lambda kv: -kv[1])
        needs = data["answers"].get("needs_tool", {}).get("noul") if self.ask_needs_tool else None
        return Decision(tool=tool, confidence=confidence, ranked=ranked, needs_tool=needs,
                        usage=data.get("usage") or {})

    def rerank(self, query: str, candidates: Sequence["Tool"]) -> Decision:
        body = json.dumps(self.build_request(query, candidates)).encode()
        req = urllib.request.Request(self.url, data=body, method="POST", headers={
            "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.load(resp)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            raise RuntimeError(f"Jev request failed with HTTP {e.code}: {detail}") from e
        decision = self.parse_response(data, candidates)
        decision.latency_s = time.perf_counter() - t0
        return decision


class CallableReranker:
    """Wrap any ``(query, candidates) -> Decision`` function (tests, custom pickers)."""

    def __init__(self, fn, name: str = "callable"):
        self._fn, self.name = fn, name

    def rerank(self, query: str, candidates: Sequence["Tool"]) -> Decision:
        return self._fn(query, candidates)
