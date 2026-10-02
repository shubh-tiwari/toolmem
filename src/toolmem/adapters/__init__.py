"""Use a ``ToolRegistry`` inside agent frameworks.

Each framework lives in its own module so importing ``toolmem`` never imports a framework::

    from toolmem.adapters.langchain import to_langchain_tools
    from toolmem.adapters.llamaindex import to_llamaindex_tools
    from toolmem.adapters.openai_agents import to_openai_agents_tools

Every module offers the same two patterns: ``to_<framework>_tools(registry, query, k)`` turns the
tools retrieved for one request into native tool objects, and ``search_tools_<framework>(registry)``
is a native ``search_tools`` tool the agent can call to discover tools itself. Native tools run
through ``registry.call``, so Python functions and MCP tools loaded with ``call=True`` both work.
"""
from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, Callable, Sequence

if TYPE_CHECKING:
    from ..registry import ToolRegistry
    from ..tool import Tool

SEARCH_TOOL_DESCRIPTION = ("Search the tool catalog for tools that can help with a task. Call this when none of "
                           "your current tools fit. Returns the names, descriptions and parameters of matching tools.")
SEARCH_TOOL_SCHEMA = {"type": "object", "properties": {
    "query": {"type": "string", "description": "Plain-language description of the task."},
    "k": {"type": "integer", "description": "How many tools to return.", "default": 5}},
    "required": ["query"]}


def require(module: str, extra: str):
    """Import a framework module or explain which extra installs it."""
    import importlib
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise ImportError(f"{module} is not installed; run `pip install -e '.[{extra}]'` in your toolmem checkout") from e


def retrieve(registry: "ToolRegistry", query: str, k: int = 20, **search_kw) -> list["Tool"]:
    return [r.tool for r in registry.search(query, k=k, **search_kw)]


def by_names(registry: "ToolRegistry", names: Sequence[str]) -> list["Tool"]:
    """Registry tools for the given names, skipping unknown ones (an agent may misspell a name)."""
    return [registry.get(n) for n in dict.fromkeys(names) if n in registry]


def runner(tool: "Tool", registry: "ToolRegistry | None") -> Callable[..., Any]:
    """A ``**kwargs -> result`` callable that executes ``tool``, via the registry when given."""
    if registry is not None:
        return lambda **kw: registry.call(tool.name, kw)
    return lambda **kw: tool(**kw)


async def arun(fn: Callable[..., Any], **kw) -> Any:
    """Run a blocking tool call in a thread, so async agents are not stalled by it."""
    return await asyncio.to_thread(fn, **kw)


def search_results(registry: "ToolRegistry", query: str, k: int = 5) -> str:
    """JSON list of ``{name, description, parameters}`` for the tools retrieved for ``query``."""
    tools = retrieve(registry, query, k=int(k))
    return json.dumps([{"name": t.name, "description": t.description, "parameters": t.parameters} for t in tools])


def to_text(result: Any) -> str:
    """Tool results as text, for frameworks that hand them straight back to the model."""
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, default=str)
    except (TypeError, ValueError):
        return str(result)
