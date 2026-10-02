"""MCP loader tests against a local stdio server (no network)."""
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from toolmem import HashingEmbedder, ToolRegistry  # noqa: E402
from toolmem.mcp_loader import (MCPServerConfig, add_mcp_server, clean_name, clean_schema,  # noqa: E402
                                load_mcp_tools, to_rows)

ROOT = Path(__file__).resolve().parents[1]
SERVER = MCPServerConfig("echo", sys.executable, [str(Path(__file__).parent / "mcp_echo_server.py")],
                         package="local-echo")


def test_clean_name_and_schema():
    assert clean_name("github", "create_issue") == "github__create_issue"
    assert clean_name("x", "a.b c") == "x__a_b_c"
    assert clean_name("x", "a.b", prefix=False) == "a_b"
    assert len(clean_name("s", "t" * 100)) == 64
    assert clean_schema({"$schema": "http://json-schema.org/draft-07/schema#", "type": "object",
                         "properties": {"q": {"type": "string"}}}) == {"type": "object", "properties": {"q": {"type": "string"}}}
    assert clean_schema(None) == {"type": "object", "properties": {}}
    assert clean_schema({"type": "string"}) == {"type": "object", "properties": {}}


def test_load_tools_from_stdio_server():
    tools = {t.name: t for t in load_mcp_tools(SERVER, timeout=60)}
    assert set(tools) == {"echo__echo", "echo__add_numbers", "echo__get_weather"}
    t = tools["echo__add_numbers"]
    assert t.metadata["original_name"] == "add.numbers" and t.metadata["server"] == "echo"
    assert t.metadata["package"] == "local-echo" and t.tags == ["echo"]
    assert t.parameters["type"] == "object" and set(t.parameters["properties"]) == {"a", "b"}
    assert "$schema" not in t.parameters
    assert t.func is None


def test_registry_search_and_call():
    reg = ToolRegistry(embedder=HashingEmbedder())
    add_mcp_server(reg, SERVER, timeout=60, call=True)
    assert reg.search("weather forecast for Paris", k=1)[0].name == "echo__get_weather"
    assert reg.call("echo__echo", {"message": "hi", "shout": True}) == "HI"
    assert reg.call("echo__add_numbers", '{"a": 2, "b": 3}') == "5"  # original dotted name is used


def test_sync_call_inside_event_loop_is_refused():
    import asyncio

    async def inner():
        with pytest.raises(RuntimeError, match="running event loop"):
            load_mcp_tools(SERVER)

    asyncio.run(inner())


def test_collect_tools_rows_unchanged():
    """The refactored benchmark script must rebuild data/tools.json rows exactly."""
    sys.path.insert(0, str(ROOT / "benchmark"))
    import collect_tools
    from servers import SERVERS

    rows = json.load(open(ROOT / "benchmark" / "data" / "tools.json"))
    by_server = {s["name"]: s for s in SERVERS}
    for row in rows[:40]:
        m = row["metadata"]
        raw = {"name": m["original_name"], "description": row["description"], "inputSchema": row["parameters"]}
        rebuilt = collect_tools.to_rows(by_server[m["server"]], [raw])[0]
        rebuilt["metadata"]["collected_at"] = m["collected_at"]
        assert json.dumps(rebuilt) == json.dumps(row)
    assert to_rows({"name": "s", "command": "x"}, [{"name": "t"}])[0]["metadata"].keys() == {
        "server", "original_name", "collected_at"}


def test_load_server_configs_formats(tmp_path):
    from toolmem.mcp_loader import load_server_configs
    claude = {"mcpServers": {"gh": {"command": "npx", "args": ["-y", "x"], "env": {"T": "1"}},
                             "remote": {"url": "https://example.com/mcp"}}}
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps(claude))
    (cfg,) = load_server_configs(path)  # url-only servers are skipped
    assert (cfg.name, cfg.command, cfg.args, cfg.env) == ("gh", "npx", ["-y", "x"], {"T": "1"})
    assert [c.name for c in load_server_configs({"a": {"command": "uvx"}})] == ["a"]
    assert [c.name for c in load_server_configs([{"name": "b", "command": "uvx"}])] == ["b"]
