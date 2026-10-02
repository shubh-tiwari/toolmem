"""Framework adapters: schemas pass through, and native invocation runs the registry tool. No model calls."""
import asyncio
import json

import pytest

from toolmem import HashingEmbedder, ToolRegistry

CALLS: list[tuple[str, dict]] = []


def make_registry() -> ToolRegistry:
    reg = ToolRegistry(embedder=HashingEmbedder())

    @reg.register
    def get_weather(city: str, units: str = "metric") -> dict:
        """Current weather for a city.

        Args:
            city: City name.
            units: metric or imperial.
        """
        CALLS.append(("get_weather", {"city": city, "units": units}))
        return {"city": city, "temp": 21, "units": units}

    @reg.register
    def send_email(to: list[str], subject: str, body: str = "") -> str:
        """Send an email message to one or more recipients."""
        CALLS.append(("send_email", {"to": to, "subject": subject, "body": body}))
        return f"sent to {len(to)}"

    @reg.register
    def convert_currency(amount: float, src: str, dst: str) -> float:
        """Convert an amount of money between currencies."""
        CALLS.append(("convert_currency", {"amount": amount, "src": src, "dst": dst}))
        return amount * 2
    return reg


@pytest.fixture
def reg():
    CALLS.clear()
    return make_registry()


def test_langchain(reg):
    pytest.importorskip("langchain_core")
    from toolmem.adapters.langchain import langchain_tools_by_name, search_tools_langchain, to_langchain_tools
    tools = to_langchain_tools(reg, "what's the weather in Paris", k=2)
    assert len(tools) == 2 and tools[0].name == "get_weather"
    assert tools[0].description == reg.get("get_weather").description
    assert tools[0].args_schema == reg.get("get_weather").parameters  # passed through unchanged
    assert tools[0].invoke({"city": "Paris"})["city"] == "Paris"
    (email,) = langchain_tools_by_name(reg, ["send_email", "no_such_tool"])
    assert email.invoke({"to": ["a@x.com", "b@x.com"], "subject": "hi"}) == "sent to 2"
    assert asyncio.run(email.ainvoke({"to": ["a@x.com"], "subject": "s", "body": "b"})) == "sent to 1"
    assert CALLS[1] == ("send_email", {"to": ["a@x.com", "b@x.com"], "subject": "hi", "body": ""})
    found = json.loads(search_tools_langchain(reg).invoke({"query": "convert dollars to euros", "k": 1}))
    assert found[0]["name"] == "convert_currency" and found[0]["parameters"]["required"] == ["amount", "src", "dst"]


def test_llamaindex(reg):
    pytest.importorskip("llama_index.core")
    from toolmem.adapters.llamaindex import llamaindex_tools_by_name, search_tools_llamaindex, to_llamaindex_tools
    tools = to_llamaindex_tools(reg, "what's the weather in Paris", k=2)
    weather = tools[0]
    assert weather.metadata.name == "get_weather"
    assert weather.metadata.get_parameters_dict() == reg.get("get_weather").parameters
    assert json.loads(weather.call(city="Paris").content)["temp"] == 21
    (email,) = llamaindex_tools_by_name(reg, ["send_email"])
    out = asyncio.run(email.acall(to=["a@x.com"], subject="s"))
    assert out.content == "sent to 1" and CALLS[-1][1]["to"] == ["a@x.com"]
    openai_def = email.metadata.to_openai_tool()
    assert openai_def["function"]["parameters"]["properties"]["to"]["type"] == "array"
    found = json.loads(search_tools_llamaindex(reg).call(query="convert dollars to euros", k=1).content)
    assert found[0]["name"] == "convert_currency"


def test_openai_agents(reg):
    agents = pytest.importorskip("agents")
    from agents.tool_context import ToolContext

    from toolmem.adapters.openai_agents import (search_tools_openai_agents, to_openai_agents_tools,
                                                with_tools)
    tools = to_openai_agents_tools(reg, "what's the weather in Paris", k=2)
    weather = tools[0]
    assert weather.name == "get_weather" and weather.strict_json_schema is False
    assert weather.params_json_schema == reg.get("get_weather").parameters

    def ctx(name, args):
        return ToolContext(context=None, tool_name=name, tool_call_id="call_1", tool_arguments=args)

    args = json.dumps({"city": "Paris", "units": "imperial"})
    out = asyncio.run(weather.on_invoke_tool(ctx("get_weather", args), args))
    assert json.loads(out)["units"] == "imperial" and CALLS[-1] == ("get_weather", {"city": "Paris", "units": "imperial"})

    search = search_tools_openai_agents(reg)
    args = json.dumps({"query": "email my team", "k": 1})
    found = json.loads(asyncio.run(search.on_invoke_tool(ctx("search_tools", args), args)))
    assert found[0]["name"] == "send_email"

    agent = agents.Agent(name="assistant", tools=[search])
    grown = with_tools(agent, reg, [found[0]["name"], "send_email", "no_such_tool"])
    assert [t.name for t in grown.tools] == ["search_tools", "send_email"] and len(agent.tools) == 1


def test_missing_framework_names_extra(monkeypatch):
    import builtins

    from toolmem.adapters import require
    real = builtins.__import__

    def fake(name, *a, **k):
        if name.startswith("not_a_framework"):
            raise ImportError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(ImportError, match=r"toolmem\[langchain\]"):
        require("not_a_framework.tools", "langchain")
