"""MCP proxy tests: the proxy runs as a real stdio subprocess in front of two tiny test servers."""
import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")
from mcp import ClientSession, types  # noqa: E402
from mcp.client.stdio import StdioServerParameters, stdio_client  # noqa: E402

HERE = Path(__file__).parent


def write_config(tmp_path: Path, extra: dict | None = None) -> Path:
    servers = {"echo": {"command": sys.executable, "args": [str(HERE / "mcp_echo_server.py")]},
               "notes": {"command": sys.executable, "args": [str(HERE / "mcp_notes_server.py")]},
               **(extra or {})}
    path = tmp_path / "upstream.json"
    path.write_text(json.dumps({"mcpServers": servers}))
    return path


async def with_proxy(config: Path, body, *args: str, on_message=None):
    params = StdioServerParameters(command=sys.executable, args=["-m", "toolmem.mcp_proxy", str(config), *args],
                                   env=dict(os.environ))
    with open(os.devnull, "w") as devnull:
        async with asyncio.timeout(30):
            async with stdio_client(params, errlog=devnull) as (read, write):
                async with ClientSession(read, write, message_handler=on_message) as session:
                    await session.initialize()
                    return await body(session)


def text(result) -> str:
    return "\n".join(c.text for c in result.content if c.type == "text")


def test_search_mode(tmp_path):
    async def body(s):
        listed = sorted(t.name for t in (await s.list_tools()).tools)
        found = json.loads(text(await s.call_tool("search_tools", {"query": "weather forecast for a city", "k": 3})))
        echoed = await s.call_tool("call_tool", {"name": "echo__echo", "arguments": {"message": "hi", "shout": True}})
        unknown = await s.call_tool("call_tool", {"name": "nope__nothing", "arguments": {}})
        failed = await s.call_tool("call_tool", {"name": "notes__broken"})
        return listed, found, echoed, unknown, failed

    listed, found, echoed, unknown, failed = asyncio.run(with_proxy(write_config(tmp_path), body))
    assert listed == ["call_tool", "search_tools"]
    assert found[0]["name"] == "echo__get_weather" and len(found) == 3
    assert set(found[0]) == {"name", "description", "inputSchema"}
    assert not echoed.is_error and text(echoed) == "HI"
    assert unknown.is_error and "Unknown tool 'nope__nothing'" in text(unknown)
    assert failed.is_error and text(failed) == "Error executing tool broken"  # upstream error passed through as is


def test_dynamic_mode_lists_retrieved_tools_and_notifies(tmp_path):
    seen: list[str] = []

    async def on_message(message):
        if isinstance(message, types.ToolListChangedNotification):
            seen.append(message.method)

    async def body(s):
        before = sorted(t.name for t in (await s.list_tools()).tools)
        await s.call_tool("search_tools", {"query": "save a note", "k": 2})
        after = [t.name for t in (await s.list_tools()).tools]
        saved = await s.call_tool("notes__create_note", {"title": "todo", "body": "buy milk"})
        for _ in range(50):  # the notification may land just after the call result
            if seen:
                break
            await asyncio.sleep(0.02)
        return before, after, saved

    before, after, saved = asyncio.run(with_proxy(write_config(tmp_path), body, "--mode", "dynamic",
                                                  "--pin", "echo__echo", on_message=on_message))
    assert before == ["echo__echo", "search_tools"]
    assert after[:2] == ["search_tools", "echo__echo"] and "notes__create_note" in after and len(after) == 4
    assert text(saved) == "saved todo"
    assert seen == ["notifications/tools/list_changed"]


def test_bad_upstream_is_skipped(tmp_path):
    config = write_config(tmp_path, {"broken": {"command": "definitely-not-a-real-command-xyz"}})

    async def body(s):
        return json.loads(text(await s.call_tool("search_tools", {"query": "add two numbers", "k": 1})))

    (top,) = asyncio.run(with_proxy(config, body, "--timeout", "10"))
    assert top["name"] == "echo__add_numbers"


def test_proxy_rejects_bad_mode():
    from toolmem.mcp_proxy import ToolProxy
    with pytest.raises(ValueError):
        ToolProxy([], mode="sometimes")
