"""Collect real tool definitions from MCP servers into ``data/tools.json``.

Starts each server in ``servers.py`` over stdio, calls ``tools/list`` (nothing is executed), and
writes rows that ``ToolRegistry.load()`` accepts directly. Names are prefixed with the server
(``github__create_issue``) so identically named tools from different servers do not collide. The
MCP work is done by ``toolmem.mcp_loader``; this script adds the server list and the merge into
one file.

    python benchmark/collect_tools.py                # all servers
    python benchmark/collect_tools.py github slack   # a subset (merged into the existing file)
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from servers import SERVERS  # noqa: E402
from toolmem.mcp_loader import MCPServerConfig, clean_name, clean_schema, list_mcp_tools_async  # noqa: E402,F401
from toolmem.mcp_loader import to_rows as _to_rows  # noqa: E402

OUT = Path(__file__).parent / "data" / "tools.json"
TIMEOUT_S = 120  # npx may need to download the package on first run


async def list_tools(server: dict) -> list[dict]:
    return await list_mcp_tools_async(MCPServerConfig.from_dict(server), TIMEOUT_S)


def to_rows(server: dict, raw: list[dict]) -> list[dict]:
    return _to_rows(MCPServerConfig.from_dict(server), raw)


def main(argv: list[str]) -> None:
    wanted = set(argv)
    servers = [s for s in SERVERS if not wanted or s["name"] in wanted]
    existing: dict[str, dict] = {}
    if OUT.exists() and wanted:
        existing = {r["name"]: r for r in json.load(open(OUT)) if r["metadata"]["server"] not in wanted}
    collected: dict[str, dict] = dict(existing)
    ok, failed = [], []
    for s in servers:
        t0 = time.perf_counter()
        try:
            raw = asyncio.run(list_tools(s))
        except BaseException as e:  # noqa: BLE001 - anyio wraps failures in ExceptionGroup
            msg = str(e).splitlines()[0][:100] if str(e) else type(e).__name__
            print(f"  {s['name']:20s} FAILED ({time.perf_counter() - t0:4.0f}s): {msg}", flush=True)
            failed.append(s["name"])
            continue
        rows = to_rows(s, raw)
        for r in rows:
            collected[r["name"]] = r
        ok.append((s["name"], len(rows)))
        print(f"  {s['name']:20s} {len(rows):4d} tools ({time.perf_counter() - t0:4.0f}s)", flush=True)

    rows = sorted(collected.values(), key=lambda r: r["name"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
    print(f"\n{len(rows)} tools from {len({r['metadata']['server'] for r in rows})} servers -> {OUT}")
    if failed:
        print(f"failed: {', '.join(failed)}")


if __name__ == "__main__":
    main(sys.argv[1:])
