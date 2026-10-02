"""Draft candidate eval queries for a stratified sample of tools, for human review.

A thin CLI over ``toolmem.casegen.generate_cases``. The drafter is Claude (a different model family
from the DeepSeek augmenter used at benchmark time), it is shown the target tool plus its closest
sibling tools from the same server, and it is told to write requests the way users talk without
reusing the tool's own wording. Every row is written with ``"status": "draft"``; the reviewer sets
it to ``"ok"`` (or deletes the row) and saves the result as ``data/queries.jsonl``.

    python benchmark/draft_queries.py --n-tools 80
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from toolmem import SQLiteCache, Tool  # noqa: E402
from toolmem.casegen import generate_cases  # noqa: E402

HERE = Path(__file__).parent
DRAFTER_MODEL = "claude-opus-5-5"
DRAFTER_PRICE = (4.00, 20.00)  # USD per million input/output tokens
USAGE = {"input": 0, "output": 0, "calls": 0}
_usage_lock = threading.Lock()


def claude_drafter(client):
    """``prompt -> reply text`` through the Anthropic SDK; ``None`` when the reply was cut off."""
    def complete(prompt: str) -> str | None:
        resp = client.messages.create(model=DRAFTER_MODEL, max_tokens=1500, output_config={"effort": "low"},
                                      messages=[{"role": "user", "content": prompt}])
        with _usage_lock:
            USAGE["input"] += resp.usage.input_tokens
            USAGE["output"] += resp.usage.output_tokens
            USAGE["calls"] += 1
        if resp.stop_reason != "end_turn":
            print(f"  stop_reason={resp.stop_reason}", file=sys.stderr)
            return None
        return "".join(b.text for b in resp.content if b.type == "text")
    complete.name = DRAFTER_MODEL
    return complete


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tools", type=Path, default=HERE / "data" / "tools.json")
    ap.add_argument("--out", type=Path, default=HERE / "data" / "queries.draft.jsonl")
    ap.add_argument("--n-tools", type=int, default=80)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cache", type=Path, default=HERE / ".cache.sqlite")
    args = ap.parse_args(argv)

    import anthropic
    client = anthropic.Anthropic(max_retries=4)
    cache = SQLiteCache(args.cache)
    tools = [Tool(**d) for d in json.load(open(args.tools))]
    print(f"drafting queries for about {args.n_tools} of {len(tools)} tools with {DRAFTER_MODEL}")
    rows = generate_cases(tools, claude_drafter(client), n_tools=args.n_tools, seed=args.seed, cache=cache,
                          model=DRAFTER_MODEL, workers=args.workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    kinds = defaultdict(int)
    for r in rows:
        kinds[r["kind"]] += 1
    print(f"wrote {len(rows)} draft rows to {args.out}  ({dict(kinds)})")
    cost = (USAGE["input"] * DRAFTER_PRICE[0] + USAGE["output"] * DRAFTER_PRICE[1]) / 1e6
    print(f"{USAGE['calls']} new calls, {USAGE['input']:,} input + {USAGE['output']:,} output tokens, ${cost:.2f}")
    print("review them: set status to ok or delete the row, then save as data/queries.jsonl")


if __name__ == "__main__":
    main()
