"""``toolmem eval`` command line tests. Offline: hashing embedder, fake drafters, a local MCP server."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from toolmem import cli

ROOT = Path(__file__).resolve().parents[1]
TOOLS = [
    {"name": "get_weather", "description": "Current weather conditions for a city",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}, "tags": ["weather"]},
    {"name": "get_forecast", "description": "Multi-day weather forecast for a city",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}, "tags": ["weather"]},
    {"name": "send_email", "description": "Send an email message to a recipient",
     "parameters": {"type": "object", "properties": {"to": {"type": "string"}}}, "tags": ["mail"]},
    {"name": "convert_currency", "description": "Convert an amount between currencies",
     "parameters": {"type": "object", "properties": {"amount": {"type": "number"}}}, "tags": ["money"]},
]
CASES = [
    {"query": "is it raining in Paris right now", "expected": ["get_weather"], "status": "ok"},
    {"query": "will next week be sunny in Rome", "expected": ["get_forecast"], "status": "ok"},
    {"query": "email Bob the slides", "expected": ["send_email"], "status": "ok"},
    {"query": "how many euros is 40 dollars", "expected": ["convert_currency"], "status": "ok"},
    {"query": "dropped row", "expected": ["send_email"], "status": "dropped"},
    {"query": "draft row about money", "expected": ["convert_currency"], "status": "draft"},
]


@pytest.fixture
def files(tmp_path):
    tools = tmp_path / "tools.json"
    tools.write_text(json.dumps(TOOLS))
    cases = tmp_path / "cases.jsonl"
    cases.write_text("\n".join(json.dumps(r) for r in CASES) + "\n")
    return tools, cases, tmp_path / "cache.sqlite"


def test_check_exit_codes(files, capsys):
    tools, cases, cache = files
    base = ["eval", "check", str(cases), "--tools", str(tools), "--cache", str(cache)]
    assert cli.main(base) == 1  # a draft row is an error
    assert "status is 'draft'" in capsys.readouterr().out
    assert cli.main(base + ["--allow-draft"]) == 0
    bad = cases.parent / "bad.jsonl"
    bad.write_text(json.dumps({"query": "x", "expected": ["no_such_tool"], "status": "ok"}) + "\n")
    assert cli.main(["eval", "check", str(bad), "--tools", str(tools)]) == 1
    assert "not in catalog" in capsys.readouterr().out


def test_score_prints_table_and_writes_json(files, capsys):
    tools, cases, cache = files
    out = cases.parent / "scores.json"
    assert cli.main(["eval", "score", str(cases), "--tools", str(tools), "--cache", str(cache),
                     "--k", "1,3", "--json", str(out), "--failures", "2"]) == 0
    text = capsys.readouterr().out
    assert "4 tools, 4 cases" in text and "hit@1" in text and "hybrid" in text
    data = json.loads(out.read_text())
    assert data["n_cases"] == 4 and set(data["retrieval"]) == {"keyword", "semantic", "hybrid"}
    assert 0.0 <= data["retrieval"]["hybrid"]["hit@3"] <= 1.0
    cli.main(["eval", "score", str(cases), "--tools", str(tools), "--cache", str(cache), "--allow-draft",
              "--json", str(out)])
    assert json.loads(out.read_text())["n_cases"] == 5  # the draft row joins; the dropped one never does


def test_score_counts_also_accept(files):
    tools, _, cache = files
    cases = files[1].parent / "aa.jsonl"
    cases.write_text(json.dumps({"query": "weather in Oslo", "expected": ["send_email"],
                                 "also_accept": ["get_weather", "get_forecast"], "status": "ok"}) + "\n")
    out = cases.parent / "aa.json"
    cli.main(["eval", "score", str(cases), "--tools", str(tools), "--cache", str(cache), "--modes", "hybrid",
              "--k", "3", "--json", str(out)])
    assert json.loads(out.read_text())["retrieval"]["hybrid"]["hit@3"] == 1.0


def test_paid_runs_need_yes(files, capsys, monkeypatch):
    tools, cases, cache = files
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    code = cli.main(["eval", "score", str(cases), "--tools", str(tools), "--cache", str(cache),
                     "--augment-model", "some/model"])
    err = capsys.readouterr().err
    assert code == cli.PAID_EXIT and "4 tool descriptions to rewrite" in err and "--yes" in err
    code = cli.main(["eval", "draft", "--tools", str(tools), "--cache", str(cache), "--model", "m",
                     "--n-tools", "4", "--out", str(cases.parent / "d.jsonl")])
    assert code == cli.PAID_EXIT and "draft prompts to m" in capsys.readouterr().err
    assert not (cases.parent / "d.jsonl").exists()


def test_missing_key_is_a_clear_error(files, capsys, monkeypatch):
    tools, cases, cache = files
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert cli.main(["eval", "draft", "--tools", str(tools), "--cache", str(cache), "--model", "m",
                     "--n-tools", "4", "--yes", "--out", str(cases.parent / "d.jsonl")]) == 1
    assert "set OPENROUTER_API_KEY" in capsys.readouterr().err


def test_draft_with_fake_generator_then_cache(files, capsys, monkeypatch):
    tools, cases, cache = files
    calls = []

    class FakeGenerator:
        def __init__(self, model, **kw):
            self.name = model

        def __call__(self, prompt):
            calls.append(prompt)
            return json.dumps({"direct": [f"request number {len(calls)}"], "confusable": None, "multi": None})

    monkeypatch.setattr("toolmem.casegen.LLMCaseGenerator", FakeGenerator)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    out = cases.parent / "draft.jsonl"
    args = ["eval", "draft", "--tools", str(tools), "--cache", str(cache), "--model", "fake/model",
            "--n-tools", "4", "--out", str(out)]
    assert cli.main(args + ["--yes"]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == len(calls) == 4 and all(r["status"] == "draft" for r in rows)
    assert "review them" in capsys.readouterr().out
    # Every prompt is now cached: no --yes needed and no new calls.
    assert cli.main(args) == 0 and len(calls) == 4


def test_mcp_config_catalog(tmp_path, capsys):
    pytest.importorskip("mcp")
    cfg = tmp_path / "mcp.json"
    cfg.write_text(json.dumps({"mcpServers": {"echo": {"command": sys.executable,
                                                       "args": [str(ROOT / "tests" / "mcp_echo_server.py")]}}}))
    cases = tmp_path / "c.jsonl"
    cases.write_text(json.dumps({"query": "repeat my words back", "expected": ["echo__echo"], "status": "ok"}) + "\n")
    assert cli.main(["eval", "check", str(cases), "--mcp-config", str(cfg)]) in (0, 1)
    assert cli.main(["eval", "score", str(cases), "--mcp-config", str(cfg), "--cache",
                     str(tmp_path / "c.sqlite"), "--modes", "hybrid"]) == 0
    assert "3 tools, 1 cases" in capsys.readouterr().out


def test_catalog_required(files, capsys):
    _, cases, _ = files
    assert cli.main(["eval", "check", str(cases)]) == 1
    assert "--tools" in capsys.readouterr().err


def test_python_dash_m_help():
    r = subprocess.run([sys.executable, "-m", "toolmem", "eval", "--help"], capture_output=True, text=True,
                       cwd=ROOT)
    assert r.returncode == 0 and "draft" in r.stdout and "score" in r.stdout
