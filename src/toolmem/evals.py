"""Retrieval evals: does the registry surface the right tool for a query?"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Sequence

if TYPE_CHECKING:
    from .registry import Mode, ToolRegistry


@dataclass
class EvalCase:
    query: str
    expected: list[str]

    def __post_init__(self):
        if isinstance(self.expected, str):
            self.expected = [self.expected]


@dataclass
class EvalReport:
    label: str
    n: int
    ks: tuple[int, ...]
    hit: dict[int, float]      # any expected tool in top-k
    recall: dict[int, float]   # fraction of expected tools in top-k
    mrr: float                 # reciprocal rank of first expected tool
    failures: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"label": self.label, "n": self.n, "mrr": round(self.mrr, 4),
                **{f"hit@{k}": round(self.hit[k], 4) for k in self.ks},
                **{f"recall@{k}": round(self.recall[k], 4) for k in self.ks}}

    def __str__(self) -> str:
        return compare([self])


def load_cases(path: str) -> list[EvalCase]:
    """Load cases from JSONL with lines like {"query": "...", "expected": ["tool_a"]}."""
    with open(path, encoding="utf-8") as f:
        return [EvalCase(**json.loads(line)) for line in f if line.strip()]


def evaluate(registry: "ToolRegistry", cases: Iterable[EvalCase], ks: Sequence[int] = (1, 3, 5),
             mode: "Mode" = "hybrid", label: str | None = None) -> EvalReport:
    cases, ks = list(cases), tuple(sorted(ks))
    if not cases:
        raise ValueError("No eval cases given.")
    hit = {k: 0.0 for k in ks}
    recall = {k: 0.0 for k in ks}
    rr_total, failures = 0.0, []
    for c in cases:
        names = [r.name for r in registry.search(c.query, k=max(ks), mode=mode)]
        exp = set(c.expected)
        for k in ks:
            found = exp & set(names[:k])
            hit[k] += bool(found)
            recall[k] += len(found) / len(exp)
        rank = next((i + 1 for i, n in enumerate(names) if n in exp), None)
        rr_total += 1 / rank if rank else 0.0
        if rank != 1:
            failures.append({"query": c.query, "expected": c.expected, "got": names[:max(ks)], "rank": rank})
    n = len(cases)
    return EvalReport(label or mode, n, ks, {k: v / n for k, v in hit.items()},
                      {k: v / n for k, v in recall.items()}, rr_total / n, failures)


@dataclass
class SelectionReport:
    """End-to-end ``ToolRegistry.select`` results, including how trustworthy the confidences are."""

    label: str
    n: int
    accuracy: float
    ece: float | None                         # expected calibration error (0 = perfectly calibrated)
    bins: list[dict] = field(default_factory=list)       # reliability diagram rows
    gated: dict[float, dict] = field(default_factory=dict)  # threshold -> coverage, accuracy
    failures: list[dict] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"{self.label}: accuracy {self.accuracy:.1%} on {self.n} cases"
                 + (f", ECE {self.ece:.3f}" if self.ece is not None else "")]
        for t, g in sorted(self.gated.items()):
            acc = f"{g['accuracy']:.1%}" if g["accuracy"] is not None else "n/a"
            lines.append(f"  confidence >= {t:.2f}: covers {g['coverage']:.0%} of cases, accuracy {acc}")
        return "\n".join(lines)


def calibration(confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10) -> tuple[float, list[dict]]:
    """Expected calibration error and reliability bins for (confidence, correct) pairs."""
    n = len(confidences)
    if n == 0:
        raise ValueError("no predictions")
    bins, ece = [], 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, c in enumerate(confidences) if lo <= c < hi or (b == n_bins - 1 and c == 1.0)]
        if not idx:
            continue
        conf = sum(confidences[i] for i in idx) / len(idx)
        acc = sum(bool(correct[i]) for i in idx) / len(idx)
        ece += len(idx) / n * abs(conf - acc)
        bins.append({"range": (round(lo, 2), round(hi, 2)), "n": len(idx), "confidence": round(conf, 4),
                     "accuracy": round(acc, 4)})
    return ece, bins


def evaluate_selection(registry: "ToolRegistry", cases: Iterable[EvalCase], candidates: int = 40,
                       mode: "Mode" = "hybrid", thresholds: Sequence[float] = (0.5, 0.7, 0.9),
                       label: str | None = None) -> SelectionReport:
    """Run ``registry.select`` on every case and score the pick and its confidence.

    ``gated`` answers "if I only trust picks above this confidence, how many requests does that
    cover and how accurate are they?", which is what confidence gating relies on.
    """
    cases = list(cases)
    if not cases:
        raise ValueError("No eval cases given.")
    confs, correct, failures = [], [], []
    for c in cases:
        d = registry.select(c.query, candidates=candidates, mode=mode)
        ok = d.name in set(c.expected)
        correct.append(ok)
        confs.append(d.confidence)
        if not ok:
            failures.append({"query": c.query, "expected": c.expected, "got": d.name,
                             "confidence": d.confidence, "top3": d.top(3)})
    n = len(cases)
    have_conf = all(x is not None for x in confs)
    ece, bins = calibration(confs, correct) if have_conf else (None, [])
    gated = {}
    if have_conf:
        for t in thresholds:
            idx = [i for i, x in enumerate(confs) if x >= t]
            gated[t] = {"coverage": len(idx) / n,
                        "accuracy": sum(correct[i] for i in idx) / len(idx) if idx else None}
    return SelectionReport(label or getattr(registry.reranker, "name", "retrieval-top1"), n,
                           sum(correct) / n, ece, bins, gated, failures)


def compare(reports: Sequence[EvalReport]) -> str:
    """Render reports side by side as a plain-text table."""
    rows = [r.to_dict() for r in reports]
    cols = [c for c in rows[0] if c != "n"]
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    line = lambda r: "  ".join(str(r[c]).ljust(widths[c]) for c in cols)
    header = line({c: c for c in cols})
    return "\n".join([header, "-" * len(header), *map(line, rows)])
