"""Load tools from MCP servers into a registry.

Each server is started over stdio, asked for ``tools/list`` (following pagination), and its tools
become ``Tool`` objects named ``<server>__<tool>`` so identically named tools from different
servers do not collide. Only the stdio transport is supported.

    from toolmem import MCPServerConfig, ToolRegistry, add_mcp_server

    reg = ToolRegistry()
    add_mcp_server(reg, MCPServerConfig("github", "npx", ["-y", "@modelcontextprotocol/server-github"],
                                        env={"GITHUB_PERSONAL_ACCESS_TOKEN": "..."}), call=True)

Requires the ``mcp`` package (``pip install "toolmem[mcp]"``).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from .tool import Tool

if TYPE_CHECKING:
    from .registry import ToolRegistry

NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")
MAX_NAME_LEN = 64  # Anthropic's tool-name limit


@dataclass
class MCPServerConfig:
    """How to start one MCP server over stdio.

    ``prefix`` puts ``<name>__`` in front of every tool name (on by default, since several
    servers ship a tool called ``search``). ``env`` is added to the current environment.
    ``package`` is recorded in each tool's metadata when given.
    """

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    prefix: bool = True
    package: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "MCPServerConfig":
        """Accepts ``{"name", "command", "args", "env", "package", "prefix"}`` (the benchmark's format)."""
        return cls(name=d["name"], command=d["command"], args=list(d.get("args") or []),
                   env=dict(d.get("env") or {}), prefix=d.get("prefix", True), package=d.get("package"))


def load_server_configs(path_or_data: "str | Path | dict | list") -> list[MCPServerConfig]:
    """Read MCP server configs from a JSON file or already-parsed data.

    Accepts the ``{"mcpServers": {name: {"command", "args", "env"}}}`` layout used by Claude Desktop
    and Claude Code (servers with a ``url`` instead of a ``command`` are skipped: only stdio is
    supported), a bare ``{name: {...}}`` mapping, or a list of ``{"name", "command", ...}`` dicts.
    """
    data = path_or_data
    if isinstance(data, (str, Path)):
        with open(data, encoding="utf-8") as f:
            data = json.load(f)
    if isinstance(data, list):
        return [MCPServerConfig.from_dict(d) for d in data]
    servers = data.get("mcpServers", data)
    return [MCPServerConfig.from_dict({"name": name, **cfg}) for name, cfg in servers.items()
            if isinstance(cfg, dict) and cfg.get("command")]


def _mcp():
    try:
        import mcp  # noqa: F401
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as e:
        raise ImportError('Loading MCP servers needs the mcp package: pip install "toolmem[mcp]"') from e
    return ClientSession, StdioServerParameters, stdio_client


def clean_name(server: str, original: str, prefix: bool = True) -> str:
    """``server__tool`` with characters outside ``[a-zA-Z0-9_-]`` replaced, cut to 64 characters."""
    name = NAME_RE.sub("_", f"{server}__{original}" if prefix else original)
    if len(name) > MAX_NAME_LEN:
        print(f"    warning: truncating {name}", file=sys.stderr)
        name = name[:MAX_NAME_LEN]
    return name


def clean_schema(schema: dict | None) -> dict:
    """Make an MCP inputSchema safe for every provider: object root, no $schema key."""
    s = dict(schema or {})
    s.pop("$schema", None)
    s.setdefault("type", "object")
    if s["type"] != "object":
        s = {"type": "object", "properties": {}}
    s.setdefault("properties", {})
    return s


def to_rows(server: MCPServerConfig | dict, raw: list[dict]) -> list[dict]:
    """Raw ``tools/list`` entries -> rows that ``Tool(**row)`` and ``ToolRegistry.load`` accept."""
    if isinstance(server, dict):
        server = MCPServerConfig.from_dict(server)
    rows = []
    for t in raw:
        meta = {"server": server.name}
        if server.package is not None:
            meta["package"] = server.package
        meta.update({"original_name": t["name"], "collected_at": date.today().isoformat()})
        rows.append({
            "name": clean_name(server.name, t["name"], server.prefix),
            "description": (t.get("description") or "").strip(),
            "parameters": clean_schema(t.get("inputSchema") or t.get("input_schema")),
            "tags": [server.name],
            "metadata": meta,
        })
    return rows


def _params(server: MCPServerConfig):
    _, StdioServerParameters, _ = _mcp()
    return StdioServerParameters(command=server.command, args=server.args, env={**os.environ, **server.env})


async def list_mcp_tools_async(server: MCPServerConfig, timeout: float = 120.0) -> list[dict]:
    """Raw tool definitions (MCP JSON field names) from one server, all pages."""
    ClientSession, _, stdio_client = _mcp()
    from mcp.types import PaginatedRequestParams
    tools: list[dict] = []
    with open(os.devnull, "w") as devnull:
        async with asyncio.timeout(timeout):
            async with stdio_client(_params(server), errlog=devnull) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    cursor = None
                    while True:
                        res = await session.list_tools(params=PaginatedRequestParams(cursor=cursor) if cursor else None)
                        tools.extend(t.model_dump(by_alias=True, exclude_none=True) for t in res.tools)
                        # mcp 2.x names the field next_cursor; 1.x used nextCursor.
                        cursor = getattr(res, "next_cursor", None) or getattr(res, "nextCursor", None)
                        if not cursor:
                            break
    return tools


def _run(coro):
    """Run a coroutine from sync code; refuse clearly inside a running event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    coro.close()
    raise RuntimeError("called from inside a running event loop; use the *_async function and await it")


def _result_value(res: Any) -> Any:
    """Structured content when the server gives it, else the text parts joined.

    Python SDK servers wrap a plain return value as ``{"result": value}``; that wrapper is removed.
    """
    if getattr(res, "is_error", False) or getattr(res, "isError", False):
        text = " ".join(getattr(c, "text", "") for c in res.content or [])
        raise RuntimeError(f"MCP tool returned an error: {text}".strip())
    structured = getattr(res, "structured_content", None) or getattr(res, "structuredContent", None)
    if structured is not None:
        if isinstance(structured, dict) and list(structured) == ["result"]:
            return structured["result"]
        return structured
    parts = [c.text for c in res.content or [] if getattr(c, "type", "") == "text"]
    if len(parts) == len(res.content or []):
        return "\n".join(parts)
    return [c.model_dump(by_alias=True, exclude_none=True) for c in res.content]


async def call_mcp_tool_async(server: MCPServerConfig, original_name: str, arguments: dict | None = None,
                              timeout: float = 120.0) -> Any:
    """Start the server, call one tool by its original (unprefixed) name, and return its result."""
    ClientSession, _, stdio_client = _mcp()
    with open(os.devnull, "w") as devnull:
        async with asyncio.timeout(timeout):
            async with stdio_client(_params(server), errlog=devnull) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return _result_value(await session.call_tool(original_name, arguments or {}))


def _caller(server: MCPServerConfig, original_name: str, timeout: float) -> Callable[..., Any]:
    def call(**kwargs: Any) -> Any:
        return _run(call_mcp_tool_async(server, original_name, kwargs, timeout))
    return call


async def load_mcp_tools_async(server: MCPServerConfig | dict, timeout: float = 120.0, call: bool = False) -> list[Tool]:
    if isinstance(server, dict):
        server = MCPServerConfig.from_dict(server)
    rows = to_rows(server, await list_mcp_tools_async(server, timeout))
    return [Tool(func=_caller(server, r["metadata"]["original_name"], timeout) if call else None, **r) for r in rows]


def load_mcp_tools(server: MCPServerConfig | dict, timeout: float = 120.0, call: bool = False) -> list[Tool]:
    """Tools from one MCP server (stdio). Nothing is executed unless ``call=True`` and you call one.

    With ``call=True`` each tool gets a function, so ``ToolRegistry.call`` works: every call
    starts the server, runs the tool and shuts it down. That is simple and stateless, not fast;
    keep your own ``ClientSession`` open for high call volume or stateful servers.
    """
    return _run(load_mcp_tools_async(server, timeout, call))


def add_mcp_server(registry: "ToolRegistry", server: MCPServerConfig | dict, timeout: float = 120.0,
                   call: bool = False) -> list[Tool]:
    """Load a server's tools into ``registry`` (indexed on the next search) and return them."""
    tools = load_mcp_tools(server, timeout, call)
    registry.add_many(tools)
    return tools
