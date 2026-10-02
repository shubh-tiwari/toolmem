"""Reranker, select() and calibration tests. No network: Jev requests are built and parsed only."""
import json

import pytest

from toolmem import (CallableReranker, Decision, EvalCase, JevReranker, Tool, ToolRegistry, calibration,
                     evaluate_selection)


def tools():
    return [Tool("get_weather", "Get the weather forecast for a city.",
                 {"type": "object", "properties": {"city": {"type": "string"}}}),
            Tool("send_email", "Send an email to someone.",
                 {"type": "object", "properties": {"to": {"type": "string"}, "body": {"type": "string"}}}),
            Tool("convert_currency", "Convert money between currencies.")]


def test_jev_request_shape():
    r = JevReranker(api_key="k", ask_needs_tool=True)
    body = r.build_request("will it rain in Paris", tools())
    assert body["model"] == "jev-latest" and body["state"] == "will it rain in Paris"
    crit = body["questions"]["tool"]["criteria"]
    assert body["questions"]["tool"]["type"] == "choice"
    assert list(crit) == ["get_weather", "send_email", "convert_currency", "none"]
    assert crit["send_email"].endswith("(arguments: to, body)")
    assert body["questions"]["needs_tool"]["type"] == "noul"
    json.dumps(body)  # serializable


def test_jev_rejects_too_many_options():
    many = [Tool(f"t{i}", "x") for i in range(255)]
    with pytest.raises(ValueError, match="255"):
        JevReranker(api_key="k").build_request("q", many)


def test_jev_parse_response_native_and_cloudflare():
    r = JevReranker(api_key="k", ask_needs_tool=True)
    native = {"model": "jev-1.13.0", "usage": {"input_tokens": 210, "output_tokens": 0},
              "answers": {"tool": {"type": "choice", "choice": "get_weather", "confidence": 0.9,
                                   "probabilities": {"get_weather": 0.91, "send_email": 0.05, "none": 0.04}},
                          "needs_tool": {"type": "noul", "noul": 0.97}}}
    for data in (native, {"result": native, "success": True}):
        d = r.parse_response(data, tools())
        assert d.name == "get_weather" and d.confidence == pytest.approx(0.91)
        assert d.needs_tool == pytest.approx(0.97)
        assert d.top(2) == ["get_weather", "send_email"]
        assert d.usage["input_tokens"] == 210
    none = {"answers": {"tool": {"choice": "none", "probabilities": {"none": 0.8, "get_weather": 0.2}}}}
    assert JevReranker(api_key="k").parse_response(none, tools()).tool is None


def test_jev_needs_credentials(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(ValueError):
        JevReranker()
    with pytest.raises(ValueError, match="account_id"):
        JevReranker(api_key="k", provider="cloudflare")
    assert "accounts/abc/ai/run/typesafe/jev" in JevReranker(api_key="k", provider="cloudflare", account_id="abc").url


def test_select_with_and_without_reranker():
    reg = ToolRegistry()
    reg.add_many(tools())
    d = reg.select("weather forecast for Paris", candidates=3)
    assert d.name == "get_weather" and d.confidence is None

    seen = {}

    def pick_last(query, cands):
        seen["n"] = len(cands)
        return Decision(tool=cands[-1], confidence=0.6, ranked=[(cands[-1].name, 0.6)])

    reg.reranker = CallableReranker(pick_last)
    d = reg.select("weather forecast for Paris", candidates=2)
    assert seen["n"] == 2 and d.confidence == 0.6


def test_calibration_perfect_and_off():
    ece, bins = calibration([0.9] * 10, [True] * 9 + [False])
    assert ece == pytest.approx(0.0, abs=1e-9) and bins[0]["n"] == 10
    ece, _ = calibration([0.9] * 10, [False] * 10)
    assert ece == pytest.approx(0.9)


def test_evaluate_selection_gating():
    reg = ToolRegistry()
    reg.add_many(tools())
    answers = {"rain in Paris?": ("get_weather", 0.95), "email Bob": ("send_email", 0.55),
               "yen to euros": ("send_email", 0.4)}

    def fake(query, cands):
        name, p = answers[query]
        t = next(c for c in cands if c.name == name)
        return Decision(tool=t, confidence=p, ranked=[(name, p)])

    reg.reranker = CallableReranker(fake, name="fake")
    rep = evaluate_selection(reg, [EvalCase("rain in Paris?", ["get_weather"]), EvalCase("email Bob", ["send_email"]),
                                   EvalCase("yen to euros", ["convert_currency"])], candidates=3)
    assert rep.accuracy == pytest.approx(2 / 3)
    assert rep.gated[0.9] == {"coverage": pytest.approx(1 / 3), "accuracy": 1.0}
    assert rep.gated[0.5]["coverage"] == pytest.approx(2 / 3)
    assert rep.failures[0]["got"] == "send_email"
    assert "ECE" in str(rep)
