"""An MCP server that fronts many MCP servers and exposes only the tools a request needs.

The proxy starts every upstream server once (stdio), keeps the sessions open for its lifetime,
indexes all their tools in a ``ToolRegistry`` and serves over stdio itself. Two modes:

* ``search`` (default): the client sees two tools. ``search_tools(query, k)`` returns the
  retrieved tool definitions as JSON; ``call_tool(name, arguments)`` forwards a call to the
  upstream server that owns ``name`` and returns its result unchanged. Works with every client.
* ``dynamic``: ``tools/list`` returns ``search_tools``, any ``pinned`` tools and the tools
  retrieved by the most recent search, and each search sends ``notifications/tools/list_changed``.
  Retrieved tools can then be called directly by name. This only helps clients that refresh their
  tool list on that notification; others keep the list they saw at startup.

Register it in Claude Code or Claude Desktop like any stdio server::

    {"mcpServers": {"toolmem": {"command": "toolmem", "args": ["proxy", "upstream.json"]}}}

where ``upstream.json`` uses the same ``{"mcpServers": {...}}`` layout for the servers to front.
Logs go to stderr; stdout carries the protocol.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Sequence

from .mcp_loader import MCPServerConfig, _params, clean_schema, load_server_configs, to_rows
from .registry import ToolRegistry
from .tool import Tool

SEARCH_TOOL, CALL_TOOL = "search_tools", "call_tool"
MODES = ("search", "dynamic")


def _log(msg: str) -> None:
    print(f"toolmem proxy: {msg}", file=sys.stderr, flush=True)


def _mcp_server():
    try:
        from mcp import ClientSession, types
        from mcp.client.stdio import stdio_client
        from mcp.server.lowlevel import NotificationOptions, Server
        from mcp.server.stdio import stdio_server
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError('the MCP proxy needs the mcp package: pip install "toolmem[mcp]"') from e
    return ClientSession, types, stdio_client, NotificationOptions, Server, stdio_server


class ToolProxy:
    """Upstream sessions, the tool index and the request handlers. Use ``serve_proxy`` to run it."""

    def __init__(self, servers: Sequence[MCPServerConfig], mode: str = "search", k: int = 20,
                 registry: ToolRegistry | None = None, pinned: Sequence[str] = (), timeout: float = 120.0):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.servers, self.mode, self.k, self.timeout = list(servers), mode, k, timeout
        self.registry = registry if registry is not None else ToolRegistry()
        self.pinned = list(pinned)
        self.current: list[str] = []          # dynamic mode: names retrieved by the latest search
        self._sessions: dict[str, Any] = {}   # server name -> open ClientSession
        self._owner: dict[str, tuple[str, str]] = {}  # tool name -> (server name, original name)

    # ---------- upstream ----------
    async def connect(self, stack: AsyncExitStack) -> None:
        """Start every upstream server, list its tools and index them. Failures are logged and skipped."""
        ClientSession, types, stdio_client, *_ = _mcp_server()
        devnull = stack.enter_context(open(os.devnull, "w"))
        for server in self.servers:
            # Each upstream gets its own exit stack so a failed start cannot tear down the others.
            sub = AsyncExitStack()
            try:
                async with asyncio.timeout(self.timeout):
                    read, write = await sub.enter_async_context(stdio_client(_params(server), errlog=devnull))
                    session = await sub.enter_async_context(ClientSession(read, write))
                    await session.initialize()
                    raw, cursor = [], None
                    while True:
                        res = await session.list_tools(params=types.PaginatedRequestParams(cursor=cursor) if cursor else None)
                        raw.extend(t.model_dump(by_alias=True, exclude_none=True) for t in res.tools)
                        cursor = getattr(res, "next_cursor", None) or getattr(res, "nextCursor", None)
                        if not cursor:
                            break
            except BaseException as e:  # noqa: BLE001 - anyio wraps failures in ExceptionGroup
                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    raise
                await sub.aclose()
                _log(f"skipping {server.name}: {str(e).splitlines()[0][:200] if str(e) else type(e).__name__}")
                continue
            stack.push_async_callback(sub.aclose)
            self._sessions[server.name] = session
            for row in to_rows(server, raw):
                if row["name"] in (SEARCH_TOOL, CALL_TOOL):
                    _log(f"{server.name}: tool {row['name']!r} is shadowed by the proxy's own tool")
                    continue
                self.registry.add(Tool(**row))
                self._owner[row["name"]] = (server.name, row["metadata"]["original_name"])
            _log(f"{server.name}: {len(raw)} tools")
        self.registry.index()
        unknown = [n for n in self.pinned if n not in self.registry]
        if unknown:
            _log(f"pinned tools not found: {', '.join(unknown)}")
        _log(f"serving {len(self.registry)} tools from {len(self._sessions)} servers in {self.mode} mode")

    # ---------- handlers ----------
    def _tool_defs(self, names: Sequence[str]) -> list[dict]:
        return [self.registry.get(n).to_mcp() for n in names if n in self.registry]

    def search(self, query: str, k: int | None = None) -> list[dict]:
        hits = self.registry.search(query, k=max(1, int(k or self.k)))
        if self.mode == "dynamic":
            self.current = [r.name for r in hits]
        return self._tool_defs([r.name for r in hits])

    def listed(self) -> list[dict]:
        types = _mcp_server()[1]
        meta = [types.Tool(name=SEARCH_TOOL, description=(
                    f"Search {len(self.registry)} tools from {len(self._sessions)} connected services for the "
                    "ones that fit a task. Call this first, describing the task in plain language. "
                    + ("It returns tool definitions; run one with call_tool." if self.mode == "search" else
                       "The matching tools are then added to your tool list and can be called directly.")),
                    input_schema={"type": "object", "properties": {
                        "query": {"type": "string", "description": "Plain-language description of the task."},
                        "k": {"type": "integer", "description": "How many tools to return.", "default": self.k}},
                        "required": ["query"]})]
        if self.mode == "search":
            meta.append(types.Tool(name=CALL_TOOL, description=(
                "Run a tool found with search_tools. Pass its exact name and its arguments as an object."),
                input_schema={"type": "object", "properties": {
                    "name": {"type": "string", "description": "Tool name exactly as search_tools returned it."},
                    "arguments": {"type": "object", "description": "Arguments matching the tool's input schema."}},
                    "required": ["name"]}))
            return meta
        names = list(dict.fromkeys(self.pinned + self.current))
        return meta + [types.Tool(name=d["name"], description=d["description"], input_schema=clean_schema(d["inputSchema"]))
                       for d in self._tool_defs(names)]

    async def forward(self, name: str, arguments: dict | None) -> Any:
        types = _mcp_server()[1]
        owner = self._owner.get(name)
        if owner is None:
            hint = "use search_tools to find tool names" if self._owner else "no upstream servers are connected"
            return types.CallToolResult(content=[types.TextContent(type="text", text=f"Unknown tool {name!r}: {hint}.")],
                                        is_error=True)
        server, original = owner
        return await self._sessions[server].call_tool(original, arguments or {})

    async def handle_call(self, ctx: Any, name: str, arguments: dict | None) -> Any:
        types = _mcp_server()[1]
        args = arguments or {}
        if name == SEARCH_TOOL:
            if not str(args.get("query") or "").strip():
                return types.CallToolResult(content=[types.TextContent(type="text", text="search_tools needs a query.")],
                                            is_error=True)
            found = self.search(args["query"], args.get("k"))
            if self.mode == "dynamic":
                try:
                    await ctx.session.send_tool_list_changed()
                except Exception as e:  # noqa: BLE001 - a client without a back-channel still gets the result
                    _log(f"could not send tools/list_changed: {e}")
            return types.CallToolResult(content=[types.TextContent(type="text", text=json.dumps(found, indent=1))])
        if name == CALL_TOOL and self.mode == "search":
            inner = args.get("arguments") or {}
            if isinstance(inner, str):
                inner = json.loads(inner or "{}")
            return await self.forward(str(args.get("name") or ""), inner)
        return await self.forward(name, args)  # an upstream tool called directly by name


async def serve_proxy(servers: Sequence[MCPServerConfig | dict], *, mode: str = "search", k: int = 20,
                      registry_kwargs: dict | None = None, pinned: Sequence[str] = (), timeout: float = 120.0,
                      name: str = "toolmem") -> None:
    """Start the upstream servers and serve the proxy over stdio until the client disconnects."""
    _, types, _, NotificationOptions, Server, stdio_server = _mcp_server()
    configs = [MCPServerConfig.from_dict(s) if isinstance(s, dict) else s for s in servers]
    proxy = ToolProxy(configs, mode=mode, k=k, registry=ToolRegistry(**(registry_kwargs or {})),
                      pinned=pinned, timeout=timeout)

    async def on_list_tools(ctx, params):
        return types.ListToolsResult(tools=proxy.listed())

    async def on_call_tool(ctx, params):
        return await proxy.handle_call(ctx, params.name, params.arguments)

    server = Server(name, on_list_tools=on_list_tools, on_call_tool=on_call_tool)
    async with AsyncExitStack() as stack:
        await proxy.connect(stack)
        async with stdio_server() as (read, write):
            opts = server.create_initialization_options(NotificationOptions(tools_changed=mode == "dynamic"))
            await server.run(read, write, opts)


# --------------------------------------------------------------------------- command line
def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Proxy options, shared by ``python -m toolmem.mcp_proxy`` and ``toolmem proxy``."""
    parser.add_argument("config", type=Path, help='JSON file of upstream servers ({"mcpServers": {...}} layout)')
    parser.add_argument("--mode", choices=MODES, default="search",
                        help="search: search_tools + call_tool (any client); dynamic: retrieved tools are "
                             "listed directly, for clients that honor tools/list_changed")
    parser.add_argument("--k", type=int, default=20, help="tools returned per search (default 20)")
    parser.add_argument("--embedder", default="hashing", help="hashing (offline default) | st | openai[:model]")
    parser.add_argument("--cache", type=Path, default=None, help="SQLite cache for embeddings")
    parser.add_argument("--pin", action="append", default=[], metavar="TOOL",
                        help="dynamic mode: a tool that is always listed (repeatable, e.g. github__create_issue)")
    parser.add_argument("--timeout", type=float, default=120.0, help="seconds to wait for each upstream to start")


def make_embedder(spec: str):
    from .embedders import HashingEmbedder, OpenAIEmbedder, SentenceTransformerEmbedder
    if spec == "hashing":
        return HashingEmbedder()
    if spec == "st":
        return SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    if spec.startswith("openai"):
        return OpenAIEmbedder(spec.partition(":")[2] or "text-embedding-3-small")
    raise SystemExit(f"unknown embedder {spec!r} (use hashing, st or openai[:model])")


def run_from_args(args: argparse.Namespace) -> None:
    from .cache import SQLiteCache
    servers = load_server_configs(args.config)
    if not servers:
        raise SystemExit(f"no stdio servers found in {args.config}")
    kwargs: dict = {"embedder": make_embedder(args.embedder)}
    if args.cache:
        kwargs["cache"] = SQLiteCache(args.cache)
    asyncio.run(serve_proxy(servers, mode=args.mode, k=args.k, registry_kwargs=kwargs, pinned=args.pin,
                            timeout=args.timeout))


def run_proxy(config_path: str | Path, **kw: Any) -> None:
    """Sync entry point: read ``config_path`` and serve until the client disconnects."""
    asyncio.run(serve_proxy(load_server_configs(config_path), **kw))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m toolmem.mcp_proxy", description=__doc__.split("\n\n")[0])
    add_arguments(parser)
    run_from_args(parser.parse_args(argv))


if __name__ == "__main__":
    main()
