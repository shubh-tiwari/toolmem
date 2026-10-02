"""Lint the reviewed query set before a benchmark run (a thin CLI over ``toolmem.casegen.check_cases``).

Hard failures (exit 1): expected tool missing from the catalog, duplicate queries, rows still in
draft status (unless --allow-draft). Warnings: a query that reuses most of the tool's own wording
or its augmented description (likely leakage), servers with fewer than 3 queries, and a drafter
from the same model family as the augmenter (when both are given).

    python benchmark/check_queries.py [data/queries.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from toolmem.casegen import check_cases  # noqa: E402

HERE = Path(__file__).parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("queries", nargs="?", type=Path, default=HERE / "data" / "queries.jsonl")
    ap.add_argument("--tools", type=Path, default=HERE / "data" / "tools.json")
    ap.add_argument("--allow-draft", action="store_true")
    ap.add_argument("--overlap", type=float, default=0.6, help="warn when this share of query words appears in the tool text")
    ap.add_argument("--generator-model", help="model that drafted the queries, e.g. claude-opus-5-5")
    ap.add_argument("--augmenter-model", help="model that augments descriptions, e.g. deepseek/deepseek-v4.1-flash")
    args = ap.parse_args(argv)

    tools = json.load(open(args.tools))
    rows = [json.loads(l) for l in open(args.queries, encoding="utf-8") if l.strip()]
    report = check_cases(rows, tools, overlap=args.overlap, allow_draft=args.allow_draft,
                         generator_model=args.generator_model, augmenter_model=args.augmenter_model)
    print(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
