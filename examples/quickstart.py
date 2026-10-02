"""Run: python examples/quickstart.py

Offline by default (HashingEmbedder). Set OPENROUTER_API_KEY to also run LLM augmentation,
and install sentence-transformers to use real semantic embeddings.
"""
import os

from catalog import CASES, CATALOG
from toolmem import EvalCase, HashingEmbedder, SQLiteCache, ToolRegistry, compare, evaluate

try:
    from toolmem import SentenceTransformerEmbedder
    embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
except Exception:
    embedder = HashingEmbedder()
print(f"Embedder: {embedder.name}\n")

cases = [EvalCase(q, e) for q, e in CASES]
reg = ToolRegistry(embedder=embedder)
reg.add_many(CATALOG)

# 1. Retrieval
for r in reg.search("will it rain in London this weekend", k=3):
    print(f"{r.score:.4f}  {r.name}")

# 2. Compare retrieval modes
reports = [evaluate(reg, cases, mode=m, label=m) for m in ("keyword", "semantic", "hybrid")]

# 3. Optional: LLM augmentation via OpenRouter (cached, so reruns are free)
if os.environ.get("OPENROUTER_API_KEY"):
    from toolmem import LLMAugmenter
    import copy
    cache = SQLiteCache(".toolmem_cache.sqlite")
    aug = LLMAugmenter(model=os.environ.get("TOOLMEM_AUG_MODEL", "deepseek/deepseek-v4.1-flash"),
                       base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"],
                       extra_body={"reasoning": {"enabled": False}}, cache=cache)
    reg_aug = ToolRegistry(embedder=embedder, augmenter=aug, cache=cache)
    reg_aug.add_many(copy.deepcopy(CATALOG))
    reports.append(evaluate(reg_aug, cases, mode="hybrid", label="hybrid+augmented"))

print("\n" + compare(reports))
print("\nMisses (hybrid):")
for f in reports[2].failures:
    print(f"  {f['query']!r}: expected {f['expected']}, got {f['got'][:3]}")
