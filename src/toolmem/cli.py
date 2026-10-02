"""Command line: ``toolmem eval draft|check|score`` against your own tool catalog, and
``toolmem proxy`` to serve one MCP endpoint in front of many servers.

    toolmem eval draft --tools tools.json --model anthropic/claude-opus-5.5 --out cases.draft.jsonl
    toolmem eval check cases.jsonl --tools tools.json
    toolmem eval score cases.jsonl --tools tools.json --k 1,3,5,10
    toolmem proxy mcp.json --k 20

The catalog comes from ``--tools`` (a JSON list of tool dicts, the ``ToolRegistry.save`` format)
and/or ``--mcp-config`` (a Claude-style ``{"mcpServers": {...}}`` file; each server is started
once to list its tools). Commands that would make paid calls print what they would send and stop
unless ``--yes`` is given. Retrieval defaults to the free, offline ``hashing`` embedder.

Subcommands register themselves through ``COMMANDS``: each entry takes the top-level subparsers
object and adds its own parser with ``set_defaults(func=...)``, where ``func(args) -> int``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from typing import Callable

from .tool import Tool

OPENROUTER_URL = "https://openrouter.ai/api/v1"
PAID_EXIT = 2  # exit code when a paid command stops for lack of --yes


class CLIError(Exception):
    """A user-facing error: printed without a traceback, exit code 1."""


# --------------------------------------------------------------------------- inputs
def load_catalog(tools_path: str | None, mcp_config: str | None) -> list[Tool]:
    if not tools_path and not mcp_config:
        raise CLIError("give the catalog with --tools PATH and/or --mcp-config PATH")
    tools: dict[str, Tool] = {}
    if tools_path:
        with open(tools_path, encoding="utf-8") as f:
            for d in json.load(f):
                tools[d["name"]] = Tool(**d)
    if mcp_config:
        from .mcp_loader import load_mcp_tools, load_server_configs
        for server in load_server_configs(mcp_config):
            try:
                loaded = load_mcp_tools(server)
            except ImportError:
                raise
            except BaseException as e:  # noqa: BLE001 - anyio wraps failures in ExceptionGroup
                print(f"warning: could not list tools from MCP server {server.name!r}: "
                      f"{str(e).splitlines()[0] if str(e) else type(e).__name__}", file=sys.stderr)
                continue
            for t in loaded:
                tools[t.name] = t
    if not tools:
        raise CLIError("the catalog is empty")
    return list(tools.values())


def load_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def api_key(env: str | None, base_url: str | None) -> str:
    env = env or ("OPENROUTER_API_KEY" if (base_url or OPENROUTER_URL) == OPENROUTER_URL else "OPENAI_API_KEY")
    key = os.environ.get(env)
    if not key:
        raise CLIError(f"set {env} (or pick another variable with --api-key-env)")
    return key


def needs_yes(args, what: list[str]) -> bool:
    """Print what a run would send. True when it must stop because --yes was not given."""
    if not what:
        return False
    print("this run makes paid calls: " + "; ".join(what), file=sys.stderr)
    if args.yes:
        return False
    print("re-run with --yes to proceed", file=sys.stderr)
    return True


# --------------------------------------------------------------------------- draft
def cmd_draft(args) -> int:
    from .cache import SQLiteCache
    from .casegen import LLMCaseGenerator, build_prompt, generate_cases, group_of, sample_tools, siblings_for

    tools = load_catalog(args.tools, args.mcp_config)
    cache = SQLiteCache(args.cache)
    by_group: dict[str, list[Tool]] = defaultdict(list)
    for t in tools:
        by_group[group_of(t)].append(t)
    targets = sample_tools(tools, args.n_tools, random.Random(args.seed))
    todo = sum(cache.get_text("draft", args.model, hashlib.sha256(
        build_prompt(t, siblings_for(t, by_group[group_of(t)])).encode()).hexdigest()) is None for t in targets)
    if needs_yes(args, [f"{todo} of {len(targets)} draft prompts to {args.model} (the rest are cached)"] if todo else []):
        return PAID_EXIT
    complete: Callable[[str], str | None]
    if todo:
        extra = {"reasoning": {"enabled": False}} if (args.base_url or OPENROUTER_URL) == OPENROUTER_URL else None
        complete = LLMCaseGenerator(args.model, base_url=args.base_url or OPENROUTER_URL,
                                    api_key=api_key(args.api_key_env, args.base_url), extra_body=extra)
    else:
        def complete(prompt: str) -> str | None:  # every prompt is cached; never called
            raise AssertionError("unexpected model call")
    rows = generate_cases(tools, complete, n_tools=args.n_tools, seed=args.seed, cache=cache, model=args.model)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    kinds: dict[str, int] = defaultdict(int)
    for r in rows:
        kinds[r["kind"]] += 1
    print(f"wrote {len(rows)} draft rows for {len(targets)} tools to {args.out}  ({dict(kinds)})")
    print("review them before scoring: set each row's status to ok or dropped, fix wording, and list "
          "other equally correct tools in also_accept")
    return 0


# --------------------------------------------------------------------------- check
def cmd_check(args) -> int:
    from .casegen import check_cases
    report = check_cases(load_rows(args.cases), load_catalog(args.tools, args.mcp_config), overlap=args.overlap,
                         allow_draft=args.allow_draft, generator_model=args.generator_model,
                         augmenter_model=args.augmenter_model)
    print(report)
    return 0 if report.ok else 1


# --------------------------------------------------------------------------- score
def make_embedder(spec: str):
    from .embedders import HashingEmbedder, OpenAIEmbedder, SentenceTransformerEmbedder
    kind, _, model = spec.partition(":")
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "st":
        return SentenceTransformerEmbedder(model or "all-MiniLM-L6-v2")
    if kind == "openai":
        api_key("OPENAI_API_KEY", "openai")
        return OpenAIEmbedder(model or "text-embedding-3-small")
    raise CLIError(f"unknown embedder {spec!r}: use hashing, st[:model] or openai[:model]")


def score_cases(rows: list[dict], allow_draft: bool) -> list:
    from .evals import EvalCase
    cases = []
    for r in rows:
        status = r.get("status", "ok")
        if status == "dropped" or (status != "ok" and not allow_draft):
            continue
        exp = r["expected"] if isinstance(r["expected"], list) else [r["expected"]]
        cases.append(EvalCase(r["query"], exp + list(r.get("also_accept") or [])))
    return cases


def cmd_score(args) -> int:
    from .augment import DEFAULT_PROMPT, LLMAugmenter
    from .cache import SQLiteCache
    from .evals import compare, evaluate, evaluate_selection
    from .registry import ToolRegistry

    tools = load_catalog(args.tools, args.mcp_config)
    cases = score_cases(load_rows(args.cases), args.allow_draft)
    if not cases:
        raise CLIError("no scorable cases (rows need status ok; use --allow-draft for drafts)")
    ks = tuple(int(k) for k in args.k.split(","))
    modes = [m for m in args.modes.split(",") if m]
    cache = SQLiteCache(args.cache)

    paid = []
    to_augment = []
    if args.augment_model:
        aug_name = f"llm:{args.augment_model}:{hashlib.sha256(DEFAULT_PROMPT.encode()).hexdigest()[:8]}"
        to_augment = [t for t in tools if not t.augmented_description
                      and cache.get_text("augment", aug_name, t.content_hash()) is None]
        if to_augment:
            paid.append(f"{len(to_augment)} tool descriptions to rewrite with {args.augment_model}")
    if args.embedder.startswith("openai"):
        name = "openai:" + (args.embedder.partition(":")[2] or "text-embedding-3-small")
        pending = {t.name for t in to_augment}
        texts = [t.embedding_text() for t in tools if t.name not in pending] + [c.query for c in cases]
        n = len(pending) + sum(cache.get_embedding(name, x) is None for x in texts)
        if n:
            paid.append(f"{n} texts to embed with {name}")
    if args.select and args.reranker == "jev":
        paid.append(f"{len(cases)} Jev requests")
    if needs_yes(args, paid):
        return PAID_EXIT

    augmenter = None
    if args.augment_model:
        extra = {"reasoning": {"enabled": False}} if (args.base_url or OPENROUTER_URL) == OPENROUTER_URL else None
        augmenter = LLMAugmenter(args.augment_model, base_url=args.base_url or OPENROUTER_URL,
                                 api_key=api_key(args.api_key_env, args.base_url) if to_augment else "cached",
                                 cache=cache, extra_body=extra)
    reranker = None
    if args.select and args.reranker == "jev":
        from .rerank import JevReranker
        try:
            reranker = JevReranker()
        except ValueError as e:
            raise CLIError(f"{e} (set TYPESAFE_API_KEY)") from e
    reg = ToolRegistry(embedder=make_embedder(args.embedder), augmenter=augmenter, cache=cache, reranker=reranker)
    reg.add_many(tools)
    reg.index()

    reports = [evaluate(reg, cases, ks=ks, mode=m, label=m) for m in modes]
    print(f"{len(tools)} tools, {len(cases)} cases, embedder {reg.embedder.name}"
          + (f", augmented with {args.augment_model}" if augmenter else ""))
    print(compare(reports))
    if args.failures and reports:
        last = reports[-1]
        print(f"\nfirst {min(args.failures, len(last.failures))} of {len(last.failures)} cases where {last.label} "
              f"did not rank an expected tool first:")
        for f in last.failures[:args.failures]:
            print(f"  rank {f['rank'] or '-':>2}  {f['query']!r}\n           expected {f['expected']}, got {f['got'][:3]}")
    out = {"n_tools": len(tools), "n_cases": len(cases), "embedder": reg.embedder.name,
           "augment_model": args.augment_model, "retrieval": {r.label: r.to_dict() for r in reports}}
    if args.select:
        sel = evaluate_selection(reg, cases, candidates=args.candidates)
        print()
        print(sel)
        out["selection"] = {"label": sel.label, "n": sel.n, "accuracy": sel.accuracy, "ece": sel.ece,
                            "gated": {str(t): g for t, g in sel.gated.items()}}
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.json}")
    return 0


# --------------------------------------------------------------------------- parser
def add_eval_command(sub) -> None:
    catalog = argparse.ArgumentParser(add_help=False)
    catalog.add_argument("--tools", help="JSON list of tool definitions (ToolRegistry.save format)")
    catalog.add_argument("--mcp-config", help='MCP config file ({"mcpServers": {...}}); stdio servers only')
    catalog.add_argument("--cache", default=".toolmem_cache.sqlite", help="SQLite cache for drafts, embeddings and rewrites")
    llm = argparse.ArgumentParser(add_help=False)
    llm.add_argument("--base-url", help=f"OpenAI-compatible endpoint (default {OPENROUTER_URL})")
    llm.add_argument("--api-key-env", help="environment variable holding the key (default OPENROUTER_API_KEY "
                                           "for OpenRouter, else OPENAI_API_KEY)")
    llm.add_argument("--yes", action="store_true", help="allow paid calls")

    ev = sub.add_parser("eval", help="draft, check and score test queries against your catalog")
    evsub = ev.add_subparsers(dest="eval_command", required=True)

    p = evsub.add_parser("draft", parents=[catalog, llm], help="draft test queries with an LLM (paid)")
    p.add_argument("--model", required=True, help="drafter model, ideally a different family from your augmenter")
    p.add_argument("--out", default="cases.draft.jsonl")
    p.add_argument("--n-tools", type=int, default=80, help="about how many tools to draft queries for")
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_draft)

    p = evsub.add_parser("check", parents=[catalog], help="lint reviewed queries for leakage and mistakes")
    p.add_argument("cases", help="JSONL test queries")
    p.add_argument("--overlap", type=float, default=0.6, help="warn when this share of query words appears in the tool text")
    p.add_argument("--allow-draft", action="store_true", help="do not fail on rows still in draft status")
    p.add_argument("--generator-model", help="model that drafted the queries (for the same-family warning)")
    p.add_argument("--augmenter-model", help="model that rewrote the descriptions (for the same-family warning)")
    p.set_defaults(func=cmd_check)

    p = evsub.add_parser("score", parents=[catalog, llm], help="measure retrieval (hit@k, MRR) on the queries")
    p.add_argument("cases", help="JSONL test queries")
    p.add_argument("--k", default="1,3,5,10", help="comma-separated cutoffs")
    p.add_argument("--modes", default="keyword,semantic,hybrid", help="comma-separated retrieval modes")
    p.add_argument("--embedder", default="hashing", help="hashing (free) | st[:model] | openai[:model]")
    p.add_argument("--augment-model", help="rewrite descriptions with this model before indexing (paid)")
    p.add_argument("--allow-draft", action="store_true", help="also score rows still in draft status")
    p.add_argument("--failures", type=int, default=10, help="how many misses of the last mode to print")
    p.add_argument("--json", help="also write the results to this file")
    p.add_argument("--select", action="store_true", help="also score end-to-end picks with a reranker")
    p.add_argument("--reranker", default="jev", choices=["jev"], help="reranker for --select")
    p.add_argument("--candidates", type=int, default=40, help="shortlist size for --select")
    p.set_defaults(func=cmd_score)


def add_proxy_command(sub) -> None:
    from . import mcp_proxy  # imports mcp only when the proxy starts
    p = sub.add_parser("proxy", help="serve one MCP endpoint in front of many MCP servers (stdio)")
    mcp_proxy.add_arguments(p)

    def run(args) -> int:
        mcp_proxy.run_from_args(args)
        return 0
    p.set_defaults(func=run)


COMMANDS: list[Callable] = [add_eval_command, add_proxy_command]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="toolmem", description="Semantic tool memory for LLM agents.")
    sub = parser.add_subparsers(dest="command", required=True)
    for register in COMMANDS:
        register(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except CLIError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except (ImportError, OSError) as e:  # missing extra, or a config/tools file that cannot be read
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
