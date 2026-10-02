"""Offline end-to-end run of the benchmark harness (FakeProvider + HashingEmbedder)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))
sys.path.insert(0, str(ROOT / "examples"))

from catalog import CASES, CATALOG  # noqa: E402

import run as bench_run  # noqa: E402


def write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    tools = tmp_path / "tools.json"
    tools.write_text(json.dumps([{"name": t.name, "description": t.description, "parameters": t.parameters,
                                  "tags": t.tags, "metadata": {"server": "demo"}} for t in CATALOG]))
    queries = tmp_path / "queries.jsonl"
    rows = [{"query": q, "expected": e, "server": "demo", "kind": "direct", "status": "ok"} for q, e in CASES]
    rows[0]["status"] = "draft"  # excluded unless --allow-draft
    queries.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return tools, queries


def test_offline_benchmark_runs(tmp_path):
    tools, queries = write_fixture(tmp_path)
    args = ["--provider", "fake", "--embedder", "hashing", "--conditions", "all@5,all@all,toolmem@3",
            "--tools", str(tools), "--queries", str(queries), "--cache", str(tmp_path / "c.sqlite"),
            "--out", str(tmp_path / "results"), "--workers", "2"]
    summary = bench_run.main(args)
    conds = summary["models"]["fake"]["conditions"]
    assert set(conds) == {"all@5", "all@all", "toolmem@3"}
    assert summary["meta"]["n_queries"] == len(CASES) - 1  # draft row skipped
    for s in conds.values():
        assert s["errors"] == 0 and 0.0 <= s["accuracy"] <= 1.0
    assert conds["all@5"]["tools_per_request"] == 5
    assert conds["all@all"]["tools_per_request"] == len(CATALOG)
    assert conds["all@all"]["retrieval_recall"] == 1.0  # every tool present
    assert conds["toolmem@3"]["tools_per_request"] == 3
    assert conds["all@all"]["mean_input_tokens"] > conds["toolmem@3"]["mean_input_tokens"]
    assert "hybrid" in summary["retrieval"]["hashing-1024"]
    runs = (tmp_path / "results" / "runs.fake.fake.jsonl").read_text().splitlines()
    assert len(runs) == 3 * (len(CASES) - 1)

    # Second run is served from cache and merges into the same summary file.
    summary2 = bench_run.main(args + ["--allow-draft"])
    assert summary2["meta"]["n_queries"] == len(CASES)
    assert set(summary2["models"]["fake"]["conditions"]) == {"all@5", "all@all", "toolmem@3"}


def test_dry_run_makes_no_results(tmp_path):
    tools, queries = write_fixture(tmp_path)
    out = bench_run.main(["--provider", "fake", "--embedder", "hashing", "--conditions", "all@all,toolmem@2",
                          "--tools", str(tools), "--queries", str(queries), "--cache", str(tmp_path / "c.sqlite"),
                          "--out", str(tmp_path / "results"), "--dry-run"])
    assert out["dry_run"] is True and out["projected_usd"] == 0.0
    assert not (tmp_path / "results").exists()


def test_subset_always_contains_expected(tmp_path):
    tools, queries = write_fixture(tmp_path)
    bench_run.main(["--provider", "fake", "--embedder", "hashing", "--conditions", "all@3",
                    "--tools", str(tools), "--queries", str(queries), "--cache", str(tmp_path / "c.sqlite"),
                    "--out", str(tmp_path / "results")])
    for line in (tmp_path / "results" / "runs.fake.fake.jsonl").read_text().splitlines():
        r = json.loads(line)
        assert r["n_tools"] == 3 and r["retrieved"] == 1


def test_limit_run_keeps_full_run_meta(tmp_path):
    tools, queries = write_fixture(tmp_path)
    base = ["--provider", "fake", "--embedder", "hashing", "--conditions", "all@all", "--tools", str(tools),
            "--queries", str(queries), "--cache", str(tmp_path / "c.sqlite"), "--out", str(tmp_path / "results")]
    full = bench_run.main(base)["meta"]["n_queries"]
    summary = bench_run.main(base + ["--limit", "2"])
    assert summary["meta"]["n_queries"] == full
    assert summary["models"]["fake"]["n_queries"] == 2


def test_target_scoring_and_billed_cost():
    rec = {"expected": ["a", "b"], "also_accept": [], "tool": "b", "correct": 1,
           "reported_cost": None, "cost_effective": 0.5}
    assert bench_run.target_correct(rec) == 0  # second step first: correct, but not target-first
    assert bench_run.target_correct({**rec, "tool": "a"}) == 1
    assert bench_run.billed_cost(rec) == 0.5
    assert bench_run.billed_cost({**rec, "reported_cost": 0.1}) == 0.1


def test_mcnemar_and_holm():
    import report
    a = {i: {"correct": 1} for i in range(10)}
    b = {i: {"correct": int(i < 2)} for i in range(10)}
    n, a_only, b_only, p = report.mcnemar(a, b)
    assert (n, a_only, b_only) == (10, 8, 0) and abs(p - 2 / 2 ** 8) < 1e-12
    assert report.holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_cascade_ignores_llm_failure_when_jev_is_confident():
    from providers import CascadeProvider, Selection
    from toolmem import Decision, Tool
    tools = [Tool(name="x", description="x"), Tool(name="y", description="y")]

    class FakeJev:
        def __init__(self, conf):
            self.conf = conf

        def rerank(self, query, cands):
            return Decision(tool=cands[0], confidence=self.conf, ranked=[("x", self.conf)], usage={"input_tokens": 100})

    class FailingLLM:
        model, prices = "llm", (1.0, 1.0, 1.0, 1.0)

        def select_tool(self, *a, **k):
            return Selection(tool=None, error="boom")

    c = object.__new__(CascadeProvider)
    c._llm, c.threshold, c.k = FailingLLM(), 0.9, 3
    c._jev = type("J", (), {"_r": FakeJev(0.95)})()
    assert c.select_tool("", "q", tools).tool == "x"          # confident: Jev's pick stands
    c._jev = type("J", (), {"_r": FakeJev(0.5)})()
    assert c.select_tool("", "q", tools).error.startswith("llm")  # unsure: the failure counts
