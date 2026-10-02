"""Eval-case generation with leakage checks.

``generate_cases`` drafts candidate eval queries for a stratified sample of tools: the drafter model
sees the target tool plus its closest siblings from the same group (MCP server or first tag) and is
told to write requests the way users talk, without reusing the tool's own wording. Every row comes
back with ``"status": "draft"``; a reviewer sets it to ``"ok"`` (or ``"dropped"``) before use.

``check_cases`` lints a reviewed set before an eval run: unknown tools, duplicates, unreviewed rows,
queries that copy the tool's text (or its augmented description), thin groups, and a drafter from
the same model family as the augmenter.

Draft with a different model family from the one that augments descriptions, so the rewritten
descriptions cannot have been tuned to the queries.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from .bm25 import BM25Index, tokenize
from .cache import SQLiteCache
from .tool import Tool

PROMPT = """You are writing evaluation queries for a benchmark that tests whether an AI agent picks the
right tool out of a large catalog. Below is the TARGET tool and its closest SIBLING tools from the
same MCP server.

Write requests the way a real user would type them to an assistant: short, concrete, with specific
names, values, dates or file paths made up as needed. Do NOT reuse the tool's name or the distinctive
words of its description; use everyday vocabulary and synonyms. Each request must be something the
target tool is the single best fit for, even with all the siblings available.

Return ONLY a JSON object of this shape (no prose, no code fences):
{{
  "direct": ["<request 1>", "<request 2>"],
  "confusable": {{"query": "<a request that a careless reader might route to the sibling named below, but which clearly needs the target>", "confused_with": "<sibling name>"}},
  "multi": {{"query": "<a request that needs the target AND one sibling, target first>", "also": "<sibling name>"}} or null
}}
Set "multi" to null unless a two-step request is natural.

TARGET:
{target}

SIBLINGS:
{siblings}"""


# --------------------------------------------------------------------------- generation
def group_of(tool: Tool) -> str:
    """The tool's group for stratified sampling and siblings: its MCP server, else its first tag."""
    return tool.metadata.get("server") or (tool.tags[0] if tool.tags else "")


def strip_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s


def sample_tools(tools: Sequence[Tool], n: int, rng: random.Random,
                 group: Callable[[Tool], str] = group_of) -> list[Tool]:
    """Proportional-to-size allocation per group, at least 2 per group where possible."""
    by_group: dict[str, list[Tool]] = defaultdict(list)
    for t in tools:
        by_group[group(t)].append(t)
    total = len(tools)
    picked: list[Tool] = []
    for _, ts in sorted(by_group.items()):
        k = max(min(2, len(ts)), round(n * len(ts) / total))
        picked.extend(rng.sample(ts, min(k, len(ts))))
    return picked


def siblings_for(target: Tool, same_group: Sequence[Tool], k: int = 8) -> list[Tool]:
    """The ``k`` tools in the target's group whose text is closest to it (BM25)."""
    idx = BM25Index()
    for t in same_group:
        if t.name != target.name:
            idx.upsert(t.name, t.base_text())
    names = [n for n, _ in idx.search(target.base_text(), k)]
    by = {t.name: t for t in same_group}
    return [by[n] for n in names]


def describe(t: Tool) -> str:
    props = t.parameters.get("properties") or {}
    params = ", ".join(f"{p}: {(s.get('description') or s.get('type') or '')[:80]}" for p, s in list(props.items())[:8])
    return f"name: {t.name}\ndescription: {t.description[:600]}\nparameters: {params or '(none)'}"


def build_prompt(target: Tool, siblings: Sequence[Tool], prompt: str = PROMPT) -> str:
    return prompt.format(target=describe(target), siblings="\n\n".join(describe(s) for s in siblings) or "(none)")


def rows_from_draft(target: Tool, siblings: Sequence[Tool], draft: dict,
                    group: Callable[[Tool], str] = group_of) -> list[dict]:
    """Turn the drafter's JSON into query rows. Sibling names the drafter invented are discarded."""
    sib_names = {s.name for s in siblings}
    base = {"server": group(target), "status": "draft",
            "target_original_name": target.metadata.get("original_name", target.name)}
    rows = [{"query": q, "expected": [target.name], "kind": "direct", **base} for q in draft.get("direct") or []]
    c = draft.get("confusable")
    if c and c.get("query"):
        cw = c.get("confused_with")
        rows.append({"query": c["query"], "expected": [target.name], "kind": "confusable",
                     "confused_with": cw if cw in sib_names else None, **base})
    m = draft.get("multi")
    if m and m.get("query") and m.get("also") in sib_names:
        rows.append({"query": m["query"], "expected": [target.name, m["also"]], "kind": "multi", **base})
    return rows


class LLMCaseGenerator:
    """Draft with any OpenAI-compatible chat model (OpenAI, OpenRouter, vLLM, Ollama...).

    Call it with a prompt to get the reply text, or ``None`` when the reply was cut off or empty.
    Pass it (or any ``prompt -> str | None`` function, e.g. one wrapping the Anthropic SDK) to
    ``generate_cases``. As with ``LLMAugmenter``, turn reasoning off through ``extra_body`` or raise
    ``max_tokens`` for reasoning models.
    """

    def __init__(self, model: str, client=None, base_url: str | None = None, api_key: str | None = None,
                 temperature: float = 0.7, max_tokens: int = 1500, extra_body: dict | None = None):
        if client is None:
            from openai import OpenAI  # lazy import
            client = OpenAI(base_url=base_url, api_key=api_key)
        self._client, self.model = client, model
        self.temperature, self.max_tokens, self.extra_body = temperature, max_tokens, extra_body
        self.name = model

    def __call__(self, prompt: str) -> str | None:
        kwargs = {"extra_body": self.extra_body} if self.extra_body else {}
        resp = self._client.chat.completions.create(
            model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}], **kwargs)
        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        return text if text and choice.finish_reason != "length" else None


def generate_cases(tools: Sequence[Tool], complete: Callable[[str], str | None], n_tools: int = 80,
                   seed: int = 7, cache: SQLiteCache | None = None, model: str | None = None,
                   workers: int = 6, prompt: str = PROMPT,
                   group: Callable[[Tool], str] = group_of) -> list[dict]:
    """Draft eval rows for a stratified sample of about ``n_tools`` tools.

    ``complete`` maps a prompt to the drafter's reply (``None`` if the call failed; failures are
    not cached). With ``cache``, replies are stored under ``model`` (default ``complete.name``) and
    the prompt, so re-running with the same tools and seed makes no model calls. Rows are ordered
    by group, then by sampled tool, and each has ``status: "draft"``.
    """
    model = model or getattr(complete, "name", None) or "callable"
    by_group: dict[str, list[Tool]] = defaultdict(list)
    for t in tools:
        by_group[group(t)].append(t)
    targets = sample_tools(tools, n_tools, random.Random(seed), group)

    def work(t: Tool) -> list[dict]:
        sibs = siblings_for(t, by_group[group(t)])
        p = build_prompt(t, sibs, prompt)
        key = ("draft", model, hashlib.sha256(p.encode()).hexdigest())
        text = cache.get_text(*key) if cache else None
        if text is None:
            text = complete(p)
            if text is None:
                warnings.warn(f"no draft for {t.name}: the model call failed or was cut off", RuntimeWarning)
                return []
            if cache:
                cache.set_text(text, *key)
        try:
            draft = json.loads(strip_fences(text))
        except json.JSONDecodeError:
            warnings.warn(f"could not parse drafter output for {t.name}", RuntimeWarning)
            return []
        return rows_from_draft(t, sibs, draft, group)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(work, targets))
    return [r for rs in results for r in rs]


# --------------------------------------------------------------------------- checks
def model_family(model: str) -> str:
    """``deepseek/deepseek-v4.1-flash`` -> ``deepseek``, ``claude-opus-5-5`` -> ``claude``.

    Also accepts augmenter names of the form ``llm:<model>:<hash>``.
    """
    if model.startswith("llm:"):
        model = model[4:].rsplit(":", 1)[0]
    m = re.match(r"[a-z]+", model.rsplit("/", 1)[-1].lower())
    return m.group(0) if m else model.lower()


@dataclass
class CheckReport:
    """Result of ``check_cases``. ``errors`` should block an eval run; ``warnings`` need a look."""

    n: int
    dropped: int
    n_tools: int
    by_kind: dict[str, int]
    by_group: dict[str, int]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def __str__(self) -> str:
        lines = [f"{self.n} queries ({self.dropped} dropped), {self.n_tools} tools",
                 f"by kind:   {self.by_kind}", f"by server: {self.by_group}"]
        lines += [f"WARN  {w}" for w in self.warnings]
        lines += [f"ERROR {e}" for e in self.errors]
        lines.append(f"{len(self.errors)} errors, {len(self.warnings)} warnings")
        return "\n".join(lines)


def _as_row(c: Any) -> dict:
    return c if isinstance(c, dict) else {"query": c.query, "expected": list(c.expected)}


def _as_tool(t: Any) -> Tool:
    return t if isinstance(t, Tool) else Tool(**t)


def _overlap(q_words: list[str], text: str) -> float:
    words = set(tokenize(text))
    return sum(w in words for w in q_words) / len(q_words)


def check_cases(cases: Iterable[dict | Any], tools: Iterable[Tool | dict], overlap: float = 0.6,
                allow_draft: bool = False, min_per_group: int = 3, generator_model: str | None = None,
                augmenter_model: str | None = None) -> CheckReport:
    """Lint eval cases (query rows as dicts, or ``EvalCase``) against the tool catalog.

    Errors: an expected or ``also_accept`` tool missing from the catalog, duplicate queries
    (case-insensitive), rows not in ``ok`` status (unless ``allow_draft``). Rows with status
    ``dropped`` are skipped. Warnings: a query whose words mostly (``overlap`` share or more) come
    from the target tool's name and description, or from its augmented description (the text
    augmented retrieval matches against); groups with fewer than ``min_per_group`` cases; and a
    drafter from the same model family as the augmenter. Line numbers count the rows kept.
    """
    catalog = {t.name: t for t in map(_as_tool, tools)}
    rows = [_as_row(c) for c in cases]
    dropped = sum(r.get("status") == "dropped" for r in rows)
    rows = [r for r in rows if r.get("status") != "dropped"]
    errors, warns = [], []
    seen: dict[str, int] = {}
    for i, r in enumerate(rows, 1):
        exp = r["expected"] if isinstance(r["expected"], list) else [r["expected"]]
        for e in exp + list(r.get("also_accept") or []):
            if e not in catalog:
                errors.append(f"line {i}: expected tool {e!r} not in catalog")
        q = r["query"].strip().lower()
        if q in seen:
            errors.append(f"line {i}: duplicate of line {seen[q]}: {r['query']!r}")
        seen.setdefault(q, i)
        if r.get("status", "ok") != "ok" and not allow_draft:
            errors.append(f"line {i}: status is {r.get('status')!r}")
        if exp and exp[0] in catalog:
            t = catalog[exp[0]]
            q_words = [w for w in tokenize(r["query"]) if len(w) > 2]
            if q_words:
                share = _overlap(q_words, f"{t.name} {t.description}")
                if share >= overlap:
                    warns.append(f"line {i}: {share:.0%} of words come from the tool text: {r['query']!r}")
                if t.augmented_description:
                    share = _overlap(q_words, t.augmented_description)
                    if share >= overlap:
                        warns.append(f"line {i}: {share:.0%} of words come from the augmented description: "
                                     f"{r['query']!r}")
    per_group = Counter(r.get("server", "?") for r in rows)
    for g, n in per_group.items():
        if n < min_per_group:
            warns.append(f"server {g!r} has only {n} queries")
    if generator_model and augmenter_model and model_family(generator_model) == model_family(augmenter_model):
        warns.append(f"drafter {generator_model!r} and augmenter {augmenter_model!r} are both "
                     f"{model_family(generator_model)} models; the augmented descriptions may echo the queries")
    return CheckReport(len(rows), dropped, len(catalog), dict(Counter(r.get("kind", "direct") for r in rows)),
                       dict(sorted(per_group.items())), errors, warns)
