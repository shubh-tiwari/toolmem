"""Eval-case generation and leakage checks. No network: the drafter is a Python function."""
import json
import random

import pytest

from toolmem import EvalCase, SQLiteCache, Tool
from toolmem.casegen import (LLMCaseGenerator, build_prompt, check_cases, generate_cases, model_family,
                             sample_tools, siblings_for, strip_fences)


def catalog() -> list[Tool]:
    tools = []
    for server, names in {"weather": ["get_forecast", "get_current", "get_alerts"],
                          "money": ["convert_currency", "get_rates"],
                          "mail": ["send_email", "list_inbox", "delete_email", "search_mail"]}.items():
        for n in names:
            tools.append(Tool(name=f"{server}__{n}", description=f"{n.replace('_', ' ')} for {server}",
                              metadata={"server": server, "original_name": n}))
    return tools


def drafter(prompt: str) -> str:
    """Answers like the real drafter, naming the first sibling listed in the prompt."""
    target = prompt.split("TARGET:\nname: ")[1].split("\n")[0]
    sibs = prompt.split("SIBLINGS:\n")[1]
    sib = sibs.split("name: ")[1].split("\n")[0] if "name: " in sibs else None
    return "```json\n" + json.dumps({
        "direct": [f"please do the thing {target} does", f"another way to ask for {target}"],
        "confusable": {"query": f"tricky {target}", "confused_with": sib},
        "multi": {"query": f"two steps {target}", "also": sib} if sib else None}) + "\n```"


def test_sampling_is_stratified_and_seeded():
    tools = catalog()
    a = sample_tools(tools, 4, random.Random(1))
    assert [t.name for t in a] == [t.name for t in sample_tools(tools, 4, random.Random(1))]
    per_server = {t.metadata["server"] for t in a}
    assert per_server == {"weather", "money", "mail"}  # every group gets at least its minimum


def test_siblings_come_from_the_same_group_only():
    tools = catalog()
    mail = [t for t in tools if t.metadata["server"] == "mail"]
    sibs = siblings_for(mail[0], mail)
    assert mail[0] not in sibs and {s.metadata["server"] for s in sibs} == {"mail"}


def test_generate_cases_rows_and_cache(tmp_path):
    tools = catalog()
    cache = SQLiteCache(tmp_path / "c.sqlite")
    calls = []

    def counting(p):
        calls.append(p)
        return drafter(p)

    rows = generate_cases(tools, counting, n_tools=9, cache=cache, model="test-model", workers=2)
    assert rows and all(r["status"] == "draft" for r in rows)
    assert {r["kind"] for r in rows} == {"direct", "confusable", "multi"}
    names = {t.name for t in tools}
    for r in rows:
        assert set(r["expected"]) <= names and r["server"] == r["expected"][0].split("__")[0]
        EvalCase(r["query"], r["expected"])  # rows load as eval cases
    multi = [r for r in rows if r["kind"] == "multi"]
    assert all(len(r["expected"]) == 2 for r in multi)

    n_calls = len(calls)
    again = generate_cases(tools, lambda p: pytest.fail("cached prompt called the model"),
                           n_tools=9, cache=cache, model="test-model")
    assert again == rows and len(calls) == n_calls


def test_invented_siblings_and_bad_json_are_dropped():
    tools = catalog()
    target, sibs = tools[0], tools[1:3]
    from toolmem.casegen import rows_from_draft
    rows = rows_from_draft(target, sibs, {"direct": ["q"], "confusable": {"query": "c", "confused_with": "nope"},
                                          "multi": {"query": "m", "also": "nope"}})
    assert [r["kind"] for r in rows] == ["direct", "confusable"] and rows[1]["confused_with"] is None
    with pytest.warns(RuntimeWarning, match="could not parse"):
        assert generate_cases(tools, lambda p: "not json", n_tools=1) == []
    with pytest.warns(RuntimeWarning, match="no draft"):
        assert generate_cases(tools, lambda p: None, n_tools=1) == []


def test_prompt_lists_target_and_siblings():
    tools = catalog()
    p = build_prompt(tools[0], tools[1:3])
    assert "TARGET:\nname: weather__get_forecast" in p and "name: weather__get_current" in p
    assert build_prompt(tools[0], []).endswith("SIBLINGS:\n(none)")
    assert strip_fences("```json\n{}\n```") == "{}"


def test_llm_case_generator_uses_openai_style_client():
    class Msg:
        content = ' {"direct": []} '

    class Choice:
        message, finish_reason = Msg(), "stop"

    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    assert kw["model"] == "m" and kw["extra_body"] == {"x": 1}
                    return type("R", (), {"choices": [Choice()]})()

    gen = LLMCaseGenerator("m", client=Client(), extra_body={"x": 1})
    assert gen("hi") == '{"direct": []}' and gen.name == "m"
    Choice.finish_reason = "length"
    assert gen("hi") is None  # cut off: not usable, not cached


def test_check_cases_errors_and_warnings():
    tools = catalog()
    tools[0].augmented_description = "Use when someone asks whether they need an umbrella tomorrow"
    rows = [
        {"query": "Will I need an umbrella tomorrow?", "expected": ["weather__get_forecast"], "server": "weather"},
        {"query": "will i need an umbrella tomorrow?", "expected": ["weather__get_forecast"], "server": "weather"},
        {"query": "email bob", "expected": ["mail__nope"], "server": "mail"},
        {"query": "convert currency for money", "expected": ["money__convert_currency"], "server": "money"},
        {"query": "draft row", "expected": ["mail__send_email"], "server": "mail", "status": "draft"},
        {"query": "gone", "expected": ["mail__send_email"], "server": "mail", "status": "dropped"},
    ]
    rep = check_cases(rows, tools)
    assert rep.n == 5 and rep.dropped == 1 and not rep.ok
    assert any("duplicate of line 1" in e for e in rep.errors)
    assert any("'mail__nope' not in catalog" in e for e in rep.errors)
    assert any("status is 'draft'" in e for e in rep.errors)
    assert any("augmented description" in w and "line 1" in w for w in rep.warnings)
    assert any("line 4" in w and "tool text" in w for w in rep.warnings)
    assert any("server 'money' has only 1" in w for w in rep.warnings)
    assert check_cases(rows[:1], tools, allow_draft=True, min_per_group=1).ok
    assert str(rep).endswith(f"{len(rep.errors)} errors, {len(rep.warnings)} warnings")


def test_check_cases_accepts_eval_cases_and_tool_dicts():
    tools = [{"name": t.name, "description": t.description, "metadata": t.metadata} for t in catalog()]
    rep = check_cases([EvalCase("what's it like outside", "weather__get_current")], tools, min_per_group=1)
    assert rep.ok and rep.n == 1


def test_same_family_drafter_and_augmenter_warns():
    assert model_family("deepseek/deepseek-v4.1-flash") == "deepseek"
    assert model_family("claude-opus-5-5") == "claude"
    assert model_family("qwen/qwen3.7-flash") == "qwen"
    assert model_family("llm:deepseek/deepseek-v4.1-flash:abcd1234") == "deepseek"
    rows = [{"query": "x y z", "expected": ["mail__send_email"], "server": "mail"}]
    warn = check_cases(rows, catalog(), min_per_group=1, generator_model="deepseek/deepseek-v4-flash",
                       augmenter_model="deepseek/deepseek-v4.1-flash").warnings
    assert any("both deepseek models" in w for w in warn)
    assert not check_cases(rows, catalog(), min_per_group=1, generator_model="claude-opus-5-5",
                           augmenter_model="deepseek/deepseek-v4.1-flash").warnings
