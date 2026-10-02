"""Run the toolmem benchmark: tool-selection accuracy, tokens, cost and latency per condition.

Conditions (comma-separated via --conditions):
  all@N          every tool in context, N = catalog size (10, 50, ..., or "all"); the subset for
                 N < all is a seeded random sample that always contains the expected tool
  toolmem@K      toolmem hybrid retrieval, top-K schemas in context
  toolmem_aug@K  same, with LLM-augmented descriptions used for retrieval
  toolmem_aug_shuffled@K  same shortlist in a seeded random order instead of best match first
                 (separates retrieval from the position of the right tool in the list)

Examples:
  python benchmark/run.py --provider fake --embedder hashing --allow-draft
  python benchmark/run.py --provider anthropic:claude-sonnet-5-5 --dry-run
  python benchmark/run.py --provider openrouter:deepseek/deepseek-v4-pro --budget-usd 5
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from providers import SYSTEM_PROMPT, Selection, make_provider  # noqa: E402
from toolmem import (EvalCase, HashingEmbedder, LLMAugmenter, OpenAIEmbedder, SQLiteCache, Tool,  # noqa: E402
                     ToolRegistry, evaluate)

HERE = Path(__file__).parent
DEFAULT_CONDITIONS = "all@10,all@25,all@50,all@100,all@200,all@all,toolmem@5,toolmem_aug@5"
OPENROUTER_URL = "https://openrouter.ai/api/v1"
DRY_RUN_SAMPLES = 10          # token-counted requests per condition
DRY_RUN_OUTPUT_TOKENS = 200   # assumed output per call (one tool call plus low-effort thinking)
EMBED_PRICE = {"openai:text-embedding-3-small": 0.02, "openai:text-embedding-3-large": 0.13}  # USD per M


@dataclass
class Case:
    query: str
    expected: list[str]
    kind: str = "direct"
    server: str = ""
    also_accept: list[str] = field(default_factory=list)  # equally valid tools; scored correct, not forced into subsets
    uid: int = 0   # line number in the queries file: keeps subsets and results stable when rows are dropped


# --------------------------------------------------------------------------- loading
def load_tools(path: Path) -> list[Tool]:
    with open(path, encoding="utf-8") as f:
        return [Tool(**d) for d in json.load(f)]


def load_queries(path: Path, allow_draft: bool = False) -> list[Case]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for uid, line in enumerate(l for l in f if l.strip()):
            d = json.loads(line)
            status = d.get("status", "ok")
            if status == "dropped" or (status != "ok" and not allow_draft):
                continue
            exp = d["expected"] if isinstance(d["expected"], list) else [d["expected"]]
            cases.append(Case(d["query"], exp, d.get("kind", "direct"), d.get("server", ""),
                              list(d.get("also_accept") or []), uid))
    return cases


def make_embedder(spec: str):
    if spec == "hashing":
        return HashingEmbedder()
    if spec == "st":
        from toolmem import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    if spec.startswith("openai"):
        _, _, model = spec.partition(":")
        return OpenAIEmbedder(model or "text-embedding-3-small")
    raise ValueError(f"unknown embedder {spec!r}")


def build_registries(tools: list[Tool], embedder, cache: SQLiteCache, need_aug: bool,
                     aug_model: str) -> tuple[ToolRegistry, ToolRegistry | None]:
    reg = ToolRegistry(embedder=embedder, cache=cache)
    reg.add_many(tools)
    reg.index()
    reg_aug = None
    if need_aug:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit("toolmem_aug needs OPENROUTER_API_KEY for the augmenter")
        aug = LLMAugmenter(model=aug_model, base_url=OPENROUTER_URL, api_key=key, cache=cache,
                           extra_body={"reasoning": {"enabled": False}})
        reg_aug = ToolRegistry(embedder=embedder, augmenter=aug, cache=cache)
        reg_aug.add_many(copy.deepcopy(tools))
        t0 = time.perf_counter()
        reg_aug.index()
        n_aug = sum(bool(t.augmented_description) for t in reg_aug)
        print(f"augmented {n_aug}/{len(tools)} tools with {aug_model} in {time.perf_counter() - t0:.1f}s (cached after first run)")
        if n_aug < 0.95 * len(tools):
            raise SystemExit(f"only {n_aug} of {len(tools)} augmentations succeeded; toolmem_aug would be invalid")
    return reg, reg_aug


def estimate_tokens(query: str, tools: list[Tool]) -> int:
    """~4 characters per token; used where the provider has no token-counting endpoint."""
    return (len(SYSTEM_PROMPT) + len(query) + sum(len(json.dumps(t.to_openai())) for t in tools)) // 4


def estimate_setup(tools: list[Tool], cases: list[Case], cache: SQLiteCache, args, need_aug: bool) -> float:
    """One-time spend a live run would add: embeddings and augmentations not yet in the cache."""
    usd = 0.0
    if args.embedder.startswith("openai"):
        name = "openai:" + (args.embedder.partition(":")[2] or "text-embedding-3-small")
        texts = [t.embedding_text() for t in tools] + [c.query for c in cases]
        new = [x for x in texts if cache.get_embedding(name, x) is None]
        tok = sum(len(x) for x in new) / 4 * (2 if need_aug else 1)  # augmented texts are about 2x longer
        usd += tok * EMBED_PRICE.get(name, 0.13) / 1e6
        print(f"setup: {len(new)} texts to embed with {name}, ~{tok:,.0f} tokens")
    if need_aug:
        from providers import openrouter_price
        from toolmem.augment import DEFAULT_PROMPT
        aug_name = f"llm:{args.aug_model}:{hashlib.sha256(DEFAULT_PROMPT.encode()).hexdigest()[:8]}"
        todo = [t for t in tools if cache.get_text("augment", aug_name, t.content_hash()) is None]
        price = openrouter_price(args.aug_model) or (1.0, 5.0, 1.0, 1.0)
        tin = sum(len(DEFAULT_PROMPT) + len(t.base_text()) for t in todo) / 4
        tout = len(todo) * 250
        usd += (tin * price[0] + tout * price[1]) / 1e6
        print(f"setup: {len(todo)} tools to augment with {args.aug_model} (${price[0]:.3f}/M in, ${price[1]:.3f}/M out)")
    return usd


# --------------------------------------------------------------------------- conditions
def parse_condition(spec: str) -> tuple[str, str]:
    kind, _, arg = spec.partition("@")
    if kind not in ("all", "toolmem", "toolmem_aug", "toolmem_shuffled", "toolmem_aug_shuffled") or not arg:
        raise ValueError(f"bad condition {spec!r}")
    return kind, arg


def tools_for_case(kind: str, arg: str, case: Case, i: int, tools: list[Tool], by_name: dict[str, Tool],
                   reg: ToolRegistry, reg_aug: ToolRegistry | None, seed: int) -> list[Tool]:
    if kind == "all":
        if arg == "all" or int(arg) >= len(tools):
            return tools
        n = int(arg)
        expected = [by_name[e] for e in case.expected if e in by_name]
        rng = random.Random(f"{seed}:{i}:{n}")
        others = [t for t in tools if t.name not in case.expected]
        picked = expected + rng.sample(others, max(0, n - len(expected)))
        rng.shuffle(picked)
        return picked
    r = reg_aug if kind.startswith("toolmem_aug") else reg
    assert r is not None
    shortlist = [res.tool for res in r.search(case.query, k=int(arg), mode="hybrid")]
    if kind.endswith("_shuffled"):
        random.Random(f"{seed}:{i}:shuffle").shuffle(shortlist)
    return shortlist


def tools_hash(tools: list[Tool]) -> str:
    return hashlib.sha256("|".join(t.content_hash() for t in tools).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- aggregation
def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, round(p / 100 * (len(xs) - 1))))
    return xs[k]


def billed_cost(r: dict) -> float:
    """What the provider charged: its reported cost when it gives one (OpenRouter, after prompt
    caching and upstream pricing), else the token count at cache-aware rates."""
    return r["reported_cost"] if r["reported_cost"] is not None else r["cost_effective"]


def target_correct(r: dict) -> int:
    """Stricter scoring for two-step queries: the first call must be the target tool, not the
    second step. Same as ``correct`` for single-tool queries."""
    return int(r["tool"] in {r["expected"][0], *r["also_accept"]}) if r["expected"] else r["correct"]


def summarize(records: list[dict], prices) -> dict:
    n = len(records)
    ok = [r for r in records if not r["error"]]
    correct = sum(r["correct"] for r in ok)
    no_tool = sum(1 for r in ok if r["tool"] is None)
    cut_off = sum(1 for r in ok if r["stop_reason"] == "length")
    billed = sum(billed_cost(r) for r in ok)
    list_cost = sum(r["cost_list"] for r in ok)
    eff_cost = sum(r["cost_effective"] for r in ok)
    lat = [r["latency_s"] for r in ok if r["latency_s"] > 0]
    by_kind: dict[str, dict] = {}
    for r in ok:
        b = by_kind.setdefault(r["kind"], {"n": 0, "correct": 0})
        b["n"] += 1
        b["correct"] += r["correct"]
    for b in by_kind.values():
        b["accuracy"] = round(b["correct"] / b["n"], 4) if b["n"] else None
    in_tok = [r["input_tokens"] + r["cache_read"] + r["cache_write"] for r in ok]
    return {
        "n": n, "errors": n - len(ok),
        "accuracy": round(correct / len(ok), 4) if ok else None,
        "target_accuracy": round(sum(target_correct(r) for r in ok) / len(ok), 4) if ok else None,
        "no_tool_rate": round(no_tool / len(ok), 4) if ok else None,
        "cut_off_rate": round(cut_off / len(ok), 4) if ok else None,
        "retrieval_recall": round(sum(r["retrieved"] for r in ok) / len(ok), 4) if ok else None,
        "tools_per_request": round(statistics.mean(r["n_tools"] for r in ok), 1) if ok else None,
        "mean_input_tokens": round(statistics.mean(in_tok)) if in_tok else None,
        "mean_output_tokens": round(statistics.mean(r["output_tokens"] for r in ok)) if ok else None,
        "mean_cache_read_tokens": round(statistics.mean(r["cache_read"] for r in ok)) if ok else None,
        "total_input_tokens": sum(in_tok),
        "cost_list_usd": round(list_cost, 4), "cost_effective_usd": round(eff_cost, 4),
        "cost_per_1k_requests_usd": round(list_cost / len(ok) * 1000, 2) if ok else None,
        "billed_cost_usd": round(billed, 4),
        "billed_per_1k_requests_usd": round(billed / len(ok) * 1000, 2) if ok else None,
        "reported_cost_usd": round(sum(r["reported_cost"] or 0 for r in ok), 4)
        if any(r["reported_cost"] is not None for r in ok) else None,
        "latency_p50_s": round(percentile(lat, 50), 2), "latency_p95_s": round(percentile(lat, 95), 2),
        "by_kind": by_kind, "prices_per_mtok": list(prices),
    }


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="fake", help="anthropic[:model] | openrouter:<model> | fake")
    ap.add_argument("--conditions", default=DEFAULT_CONDITIONS)
    ap.add_argument("--tools", type=Path, default=HERE / "data" / "tools.json")
    ap.add_argument("--queries", type=Path, default=HERE / "data" / "queries.jsonl")
    ap.add_argument("--embedder", default="openai", help="openai[:model] | st | hashing")
    ap.add_argument("--aug-model", default="deepseek/deepseek-v4.1-flash", help="OpenRouter model for augmentation")
    ap.add_argument("--effort", default="low", help="Anthropic output_config.effort")
    ap.add_argument("--cache", type=Path, default=HERE / ".cache.sqlite")
    ap.add_argument("--out", type=Path, default=HERE / "results")
    ap.add_argument("--limit", type=int, default=0, help="only the first N queries")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--budget-usd", type=float, default=0.0, help="stop issuing new calls once real spend (with caching) passes this")
    ap.add_argument("--dry-run", action="store_true", help="estimate tokens and cost, make no model calls")
    ap.add_argument("--allow-draft", action="store_true", help="include queries whose status is not ok")
    ap.add_argument("--fresh", action="store_true", help="ignore cached model responses")
    ap.add_argument("--cache-only", action="store_true", help="never call a model; uncached jobs are recorded as errors")
    args = ap.parse_args(argv)

    conditions = [parse_condition(c) for c in args.conditions.split(",") if c]
    tools = load_tools(args.tools)
    by_name = {t.name: t for t in tools}
    cases = load_queries(args.queries, args.allow_draft)
    if args.limit:
        cases = cases[:args.limit]
    missing = {e for c in cases for e in c.expected + c.also_accept if e not in by_name}
    if missing:
        raise SystemExit(f"{len(missing)} expected tools are not in the catalog: {sorted(missing)[:5]}...")
    if not cases:
        raise SystemExit("no queries loaded (use --allow-draft for unreviewed rows)")
    print(f"{len(tools)} tools, {len(cases)} queries, provider={args.provider}, embedder={args.embedder}")

    cache = SQLiteCache(args.cache)
    need_aug = any(k.startswith("toolmem_aug") for k, _ in conditions)
    if args.dry_run:
        # A dry run makes no paid calls at all: retrieval uses the free hashing embedder and the
        # un-augmented registry stands in for the augmented one. Top-5 lists have about the same
        # token size either way, and the one-time embedding/augmentation spend is estimated below.
        setup_usd = estimate_setup(tools, cases, cache, args, need_aug)
        emb_name = "openai:" + (args.embedder.partition(":")[2] or "text-embedding-3-small")
        tools_cached = args.embedder.startswith("openai") and all(
            cache.get_embedding(emb_name, t.embedding_text()) is not None for t in tools)
        if need_aug and tools_cached:
            from toolmem.augment import DEFAULT_PROMPT
            aug_name = f"llm:{args.aug_model}:{hashlib.sha256(DEFAULT_PROMPT.encode()).hexdigest()[:8]}"
            tools_cached = all(cache.get_text("augment", aug_name, t.content_hash()) for t in tools)
        if tools_cached:
            # Tool embeddings and augmentations are all cached, so real retrieval is free except for
            # embedding any new query texts (a fraction of a cent). Shortlists then match a live run.
            embedder = make_embedder(args.embedder)
            reg, reg_aug = build_registries(tools, embedder, cache, need_aug, args.aug_model)
            print(f"dry run: retrieval uses {embedder.name} from cache (new query texts are embedded)")
        else:
            embedder = HashingEmbedder()
            reg, _ = build_registries(tools, embedder, cache, False, args.aug_model)
            reg_aug = reg if need_aug else None
            print(f"dry run: retrieval uses {embedder.name} (free) as a stand-in for {args.embedder}")
    else:
        embedder = make_embedder(args.embedder)
        reg, reg_aug = build_registries(tools, embedder, cache, need_aug, args.aug_model)

        # Retrieval-only metrics (no model calls): the ceiling for the toolmem conditions.
        eval_cases = [EvalCase(c.query, c.expected + c.also_accept) for c in cases]
        retrieval = {}
        for mode in ("keyword", "semantic", "hybrid"):
            retrieval[f"{mode}"] = evaluate(reg, eval_cases, ks=(1, 3, 5, 10), mode=mode).to_dict()
        if reg_aug is not None:
            retrieval["hybrid+aug"] = evaluate(reg_aug, eval_cases, ks=(1, 3, 5, 10), mode="hybrid").to_dict()
        print("\nretrieval (embedder=%s):" % embedder.name)
        for k, v in retrieval.items():
            print(f"  {k:12s} hit@1={v['hit@1']:.3f} hit@5={v['hit@5']:.3f} recall@5={v['recall@5']:.3f} mrr={v['mrr']:.3f}")

    provider_kw = {"effort": args.effort} if args.provider.startswith("anthropic") else {}
    provider = make_provider(args.provider, **provider_kw)
    model_slug = provider.model.replace("/", "_")
    system_hash = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:12]

    def constant(kind: str, arg: str) -> bool:
        """True when every query sees the same tool list, so prompt caching pays off."""
        return kind == "all" and (arg == "all" or int(arg) >= len(tools))

    # Build every (condition, case) job up front so dry-run and live share the same tool lists.
    jobs = []
    for kind, arg in conditions:
        cond = f"{kind}@{arg}"
        for case in cases:
            tl = tools_for_case(kind, arg, case, case.uid, tools, by_name, reg, reg_aug, args.seed)
            jobs.append((cond, case.uid, case, tl, constant(kind, arg)))

    def cache_key(cond: str, case: Case, tl: list[Tool]) -> tuple[str, ...]:
        return ("bench", provider.name, provider.model, provider_kw.get("effort", ""), cond,
                tools_hash(tl), system_hash, case.query)

    if args.dry_run:
        pin, pout, pread, pwrite = provider.prices
        out_tok = DRY_RUN_OUTPUT_TOKENS
        print(f"\nprojected spend for {provider.model}: ${pin}/M in, ${pout}/M out, ${pread}/M cache read, "
              f"${pwrite}/M cache write; assumes {out_tok} output tokens per call")
        print(f"  {'condition':16s} {'new calls':>9s} {'input tok/call':>15s} {'no caching':>11s} {'expected':>9s}")
        tot_hi = tot_lo = 0.0
        for kind, arg in conditions:
            cond = f"{kind}@{arg}"
            cj = [j for j in jobs if j[0] == cond]
            todo = [j for j in cj if args.fresh or cache.get_text(*cache_key(j[0], j[2], j[3])) is None]
            n = len(todo)
            sample = todo[:DRY_RUN_SAMPLES] or cj[:DRY_RUN_SAMPLES]
            counts = [provider.count_tokens(SYSTEM_PROMPT, c.query, tl) or estimate_tokens(c.query, tl)
                      for _, _, c, tl, _ in sample]
            mean_in = statistics.mean(counts)
            no_cache = n * (mean_in * pin + out_tok * pout) / 1e6
            if hasattr(provider, "estimate_cost"):  # mixed-price pipelines estimate their own spend
                no_cache = n * statistics.mean(provider.estimate_cost(SYSTEM_PROMPT, c.query, tl)
                                               for _, _, c, tl, _ in sample)
            if constant(kind, arg) and n and provider.name != "fake":
                base = estimate_tokens(sample[0][2].query, [])
                tool_tok = mean_in - base
                first = (tool_tok * pwrite + base * pin + out_tok * pout) / 1e6  # one cache write (warm-up)
                rest = (n - 1) * (tool_tok * pread + base * pin + out_tok * pout) / 1e6
                expected = first + rest
            else:
                expected = no_cache
            tot_hi += no_cache
            tot_lo += expected
            print(f"  {cond:16s} {n:9d} {mean_in:15,.0f} {'$%.2f' % no_cache:>11s} {'$%.2f' % expected:>9s}")
        print(f"  {'model calls':16s} {'':9s} {'':15s} {'$%.2f' % tot_hi:>11s} {'$%.2f' % tot_lo:>9s}")
        print(f"  {'setup (one-time)':16s} {'':9s} {'':15s} {'':>11s} {'$%.2f' % setup_usd:>9s}")
        if provider.name == "openrouter":
            print("  OpenRouter caching depends on the upstream provider: the true cost of cached conditions is "
                  "between 'expected' and 'no caching'.")
        return {"dry_run": True, "projected_usd": tot_lo + setup_usd, "projected_usd_no_cache": tot_hi + setup_usd}

    # Live run: response cache, concurrency, cache warm-up and a budget guard on real spend.
    lock = threading.Lock()
    spent = {"usd": 0.0, "calls": 0}
    stop = threading.Event()

    def run_job(job) -> dict:
        cond, i, case, tl, repeated = job
        key = cache_key(cond, case, tl)
        hit = None if args.fresh else cache.get_text(*key)
        if hit is not None:
            sel = Selection.from_json(hit)
        elif args.cache_only:
            sel = Selection(tool=None, error="skipped: not cached (--cache-only)")
        elif stop.is_set():
            sel = Selection(tool=None, error="skipped: budget exhausted")
        else:
            sel = provider.select_tool(SYSTEM_PROMPT, case.query, tl, cache=repeated)
            if not sel.error:
                cache.set_text(sel.to_json(), *key)
            if sel.spent is not None:  # pipelines whose benchmark spend differs from their modeled cost
                real = sel.spent
            else:
                real = sel.reported_cost if sel.reported_cost is not None else sel.cost(provider.prices)[1]
            with lock:
                spent["calls"] += 1
                spent["usd"] += real
                if args.budget_usd and spent["usd"] >= args.budget_usd:
                    stop.set()
        cost_list, cost_eff = sel.cost(provider.prices)
        ok_names = set(case.expected) | set(case.also_accept)
        return {"condition": cond, "i": i, "query": case.query, "expected": case.expected, "also_accept": case.also_accept,
                "kind": case.kind, "server": case.server, "n_tools": len(tl),
                "retrieved": int(bool(ok_names & {t.name for t in tl})),
                "tool": sel.tool, "correct": int(sel.tool in ok_names), "input_tokens": sel.input_tokens,
                "output_tokens": sel.output_tokens, "cache_read": sel.cache_read, "cache_write": sel.cache_write,
                "latency_s": sel.latency_s, "stop_reason": sel.stop_reason, "text": sel.text, "error": sel.error,
                "cost_list": cost_list, "cost_effective": cost_eff, "reported_cost": sel.reported_cost}

    records: list[dict] = []
    t0 = time.perf_counter()
    # Warm the prompt cache: one request per repeated tool list runs alone before the parallel
    # requests, so they read the cached tools instead of each paying for a cache write.
    warm_idx = {next(k for k, j in enumerate(jobs) if j[0] == c) for c in {j[0] for j in jobs if j[4]}}
    for k in sorted(warm_idx):
        records.append(run_job(jobs[k]))
    rest = [j for k, j in enumerate(jobs) if k not in warm_idx]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_job, j) for j in rest]
        for n, f in enumerate(as_completed(futs), len(warm_idx) + 1):
            records.append(f.result())
            if n % 50 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} done, {spent['calls']} new calls, ${spent['usd']:.2f} spent", flush=True)
    records.sort(key=lambda r: (r["condition"], r["i"]))
    if stop.is_set():
        print(f"budget of ${args.budget_usd} reached; remaining jobs were skipped (rerun to resume)")

    args.out.mkdir(parents=True, exist_ok=True)
    runs_path = args.out / f"runs.{provider.name}.{model_slug}.jsonl"
    # Merge with earlier runs of other conditions, so running a subset never drops results.
    ran = {r["condition"] for r in records}
    kept = []
    if runs_path.exists():
        kept = [json.loads(l) for l in open(runs_path, encoding="utf-8") if l.strip()]
        kept = [r for r in kept if r["condition"] not in ran]
    with open(runs_path, "w", encoding="utf-8") as f:
        for r in sorted(kept + records, key=lambda r: (r["condition"], r["i"])):
            f.write(json.dumps(r) + "\n")

    summary_path = args.out / "summary.json"
    summary = json.load(open(summary_path)) if summary_path.exists() else {"models": {}, "retrieval": {}}
    entry = summary["models"].setdefault(provider.model, {})
    entry["provider"] = provider.name
    entry["effort"] = provider_kw.get("effort")
    entry["run_at"] = time.strftime("%Y-%m-%d %H:%M")
    entry["n_queries"] = len(cases)
    entry.setdefault("conditions", {})
    for kind, arg in conditions:
        cond = f"{kind}@{arg}"
        entry["conditions"][cond] = summarize([r for r in records if r["condition"] == cond], provider.prices)
    # A --limit smoke test must not relabel a full run's summary with its own query count.
    if args.limit and "meta" in summary:
        print(f"--limit run: kept summary meta and retrieval from the full run ({summary['meta']['n_queries']} queries)")
    else:
        summary["retrieval"][embedder.name] = retrieval
        summary["meta"] = {"n_tools": len(tools), "n_queries": len(cases), "embedder": embedder.name,
                           "aug_model": args.aug_model if need_aug else None, "seed": args.seed,
                           "system_prompt_sha": system_hash, "tools_file": str(args.tools.name),
                           "queries_file": str(args.queries.name)}
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{provider.model}  ({time.perf_counter() - t0:.0f}s, {spent['calls']} new calls, ${spent['usd']:.2f})")
    print(f"{'condition':16s} {'acc':>6s} {'no_tool':>8s} {'recall':>7s} {'tools':>6s} {'in_tok':>8s} {'$/1k':>7s} {'p50s':>6s}")
    for cond, s in entry["conditions"].items():
        print(f"{cond:16s} {s['accuracy'] or 0:6.3f} {s['no_tool_rate'] or 0:8.3f} {s['retrieval_recall'] or 0:7.3f} "
              f"{s['tools_per_request'] or 0:6.1f} {s['mean_input_tokens'] or 0:8d} {s['cost_per_1k_requests_usd'] or 0:7.2f} {s['latency_p50_s']:6.2f}")
    print(f"wrote {runs_path} and {summary_path}")
    return summary


if __name__ == "__main__":
    main()
