import json

import numpy as np
import pytest

from toolmem import (CallableAugmenter, CallableEmbedder, EvalCase, HashingEmbedder, SQLiteCache,
                     Tool, ToolRegistry, compare, evaluate)


def make_registry(**kw):
    reg = ToolRegistry(**kw)

    @reg.register(tags=["weather"])
    def get_weather(city: str, days: int = 1) -> str:
        """Get the weather forecast for a city.

        Args:
            city: City name, e.g. "London".
            days: Number of days to forecast.
        """
        return f"sunny in {city} for {days}d"

    @reg.register
    def send_email(to: str, subject: str, body: str) -> str:
        """Send an email to someone."""
        return f"sent to {to}"

    @reg.register(tags=["finance"])
    def convert_currency(amount: float, from_ccy: str, to_ccy: str) -> float:
        """Convert money between currencies."""
        return amount * 2

    return reg


def test_schema_from_function():
    t = make_registry().get("get_weather")
    props = t.parameters["properties"]
    assert props["city"] == {"type": "string", "description": 'City name, e.g. "London".'}
    assert props["days"]["type"] == "integer" and props["days"]["default"] == 1
    assert t.parameters["required"] == ["city"]
    assert t.description == "Get the weather forecast for a city."


def test_optional_and_list_annotations():
    def f(names: list[str], limit: int | None = None):
        """Do a thing."""
    p = Tool.from_function(f).parameters["properties"]
    assert p["names"] == {"type": "array", "items": {"type": "string"}}
    assert p["limit"]["type"] == "integer"


@pytest.mark.parametrize("mode", ["hybrid", "semantic", "keyword"])
def test_search_finds_tool(mode):
    reg = make_registry()
    assert reg.search("weather forecast for Paris", k=1, mode=mode)[0].name == "get_weather"


def test_tag_filter():
    res = make_registry().search("send email", k=3, tags=["finance"])
    assert [r.name for r in res] == ["convert_currency"]


def test_call_and_export():
    reg = make_registry()
    assert reg.call("get_weather", json.dumps({"city": "Delhi"})) == "sunny in Delhi for 1d"
    oa = reg.tools_for("email", k=1)[0]
    assert oa["type"] == "function" and oa["function"]["name"] == "send_email"
    an = reg.tools_for("email", k=1, format="anthropic")[0]
    assert "input_schema" in an


def test_schema_round_trips():
    t = make_registry().get("send_email")
    for back in (Tool.from_openai(t.to_openai()), Tool.from_anthropic(t.to_anthropic()),
                 Tool.from_mcp(t.to_mcp())):
        assert back.name == t.name and back.parameters == t.parameters


def test_augmentation_is_retrieval_only():
    aug = CallableAugmenter(lambda t: "rain umbrella sunshine climate" if t.name == "get_weather" else "")
    reg = make_registry(augmenter=aug)
    assert reg.search("do I need an umbrella", k=1)[0].name == "get_weather"
    # The schema the LLM sees keeps the original description.
    assert reg.get("get_weather").to_openai()["function"]["description"] == "Get the weather forecast for a city."


def test_remove_and_reindex():
    reg = make_registry()
    reg.remove("send_email")
    assert "send_email" not in [r.name for r in reg.search("email", k=3)]
    reg.add(Tool("send_email", "Send an email."))
    assert reg.search("email", k=1)[0].name == "send_email"


def test_meta_search_tool():
    reg = make_registry()
    assert reg.search_tool()["function"]["name"] == "search_tools"
    found = reg.handle_search_tool('{"query": "currency exchange", "k": 1}')
    assert found[0]["function"]["name"] == "convert_currency"


def test_embedding_cache(tmp_path):
    calls = []
    base = HashingEmbedder()

    def fn(texts):
        calls.append(len(texts))
        return base.embed(texts)

    cache = SQLiteCache(tmp_path / "c.sqlite")
    for _ in range(2):
        make_registry(embedder=CallableEmbedder(fn, "counting"), cache=cache).index()
    assert calls == [3]  # second registry fully served from cache


def test_save_load(tmp_path):
    reg = make_registry()
    reg.save(str(tmp_path / "tools.json"))
    reg2 = ToolRegistry()
    reg2.load(str(tmp_path / "tools.json"), functions={"send_email": lambda **k: "ok"})
    assert len(reg2) == 3 and reg2.call("send_email", {"to": "a", "subject": "b", "body": "c"}) == "ok"


def test_evaluate_metrics():
    reg = make_registry()
    cases = [EvalCase("weather forecast", "get_weather"), EvalCase("convert currency", ["convert_currency"])]
    rep = evaluate(reg, cases, ks=(1, 3))
    assert rep.hit[1] == 1.0 and rep.mrr == 1.0 and rep.failures == []
    assert "hit@1" in compare([rep])


def test_normalized_embeddings():
    v = HashingEmbedder().embed(["hello world", ""])
    assert np.allclose(np.linalg.norm(v[0]), 1.0)


def test_llm_augmenter_warns_on_empty_reasoning_output():
    from types import SimpleNamespace
    from toolmem import LLMAugmenter

    calls = []

    class FakeCompletions:
        def create(self, **kw):
            calls.append(kw)
            msg = SimpleNamespace(content="")
            return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="length")])

    client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    aug = LLMAugmenter(model="m", client=client, extra_body={"reasoning": {"enabled": False}})
    tool = Tool("t", "does a thing")
    with pytest.warns(RuntimeWarning, match="reasoning"):
        assert aug.augment(tool) == ""
    assert calls[0]["extra_body"] == {"reasoning": {"enabled": False}}
