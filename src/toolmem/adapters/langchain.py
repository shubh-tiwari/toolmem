"""LangChain adapter: registry tools as ``StructuredTool`` objects (``pip install 'toolmem[langchain]'``).

    tools = to_langchain_tools(reg, user_message, k=20)       # per-request retrieval
    model.bind_tools(tools)

    tools = [search_tools_langchain(reg)]                      # or let the agent search
    # after it calls search_tools, add what it found for the next turn:
    tools += langchain_tools_by_name(reg, [d["name"] for d in json.loads(result)])

The JSON Schema is passed through as ``args_schema`` unchanged, so the model sees exactly what the
tool published.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from . import SEARCH_TOOL_DESCRIPTION, SEARCH_TOOL_SCHEMA, arun, by_names, require, retrieve, runner, search_results

if TYPE_CHECKING:
    from ..registry import ToolRegistry
    from ..tool import Tool


def tool_to_langchain(tool: "Tool", registry: "ToolRegistry | None" = None) -> Any:
    """One ``Tool`` as a LangChain ``StructuredTool`` that runs via ``registry.call`` (or the tool itself)."""
    StructuredTool = require("langchain_core.tools", "langchain").StructuredTool
    fn = runner(tool, registry)

    async def afn(**kw):
        return await arun(fn, **kw)

    return StructuredTool.from_function(func=fn, coroutine=afn, name=tool.name, description=tool.description,
                                        args_schema=tool.parameters, infer_schema=False)


def to_langchain_tools(registry: "ToolRegistry", query: str, k: int = 20, **search_kw) -> list[Any]:
    """The ``k`` tools retrieved for ``query``, best match first, as LangChain tools."""
    return [tool_to_langchain(t, registry) for t in retrieve(registry, query, k, **search_kw)]


def langchain_tools_by_name(registry: "ToolRegistry", names: Sequence[str]) -> list[Any]:
    """LangChain tools for names the agent discovered through ``search_tools``."""
    return [tool_to_langchain(t, registry) for t in by_names(registry, names)]


def search_tools_langchain(registry: "ToolRegistry") -> Any:
    """A native ``search_tools`` tool returning matching tool definitions as JSON."""
    StructuredTool = require("langchain_core.tools", "langchain").StructuredTool

    def fn(query: str, k: int = 5) -> str:
        return search_results(registry, query, k)

    async def afn(query: str, k: int = 5) -> str:
        return await arun(fn, query=query, k=k)

    return StructuredTool.from_function(func=fn, coroutine=afn, name=registry.SEARCH_TOOL_NAME,
                                        description=SEARCH_TOOL_DESCRIPTION, args_schema=SEARCH_TOOL_SCHEMA,
                                        infer_schema=False)
