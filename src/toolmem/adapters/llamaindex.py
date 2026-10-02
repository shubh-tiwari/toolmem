"""LlamaIndex adapter: registry tools as ``FunctionTool`` objects (``pip install -e ".[llamaindex]"``).

    tools = to_llamaindex_tools(reg, user_message, k=20)
    agent = FunctionAgent(tools=tools, llm=...)

LlamaIndex describes parameters with a pydantic model. The model built here reports the tool's own
JSON Schema unchanged, so the LLM sees what the tool published; it does no validation of its own.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from . import (SEARCH_TOOL_DESCRIPTION, SEARCH_TOOL_SCHEMA, arun, by_names, require, retrieve, runner,
               search_results, to_text)

if TYPE_CHECKING:
    from ..registry import ToolRegistry
    from ..tool import Tool


def schema_model(name: str, schema: dict) -> type:
    """A permissive pydantic model whose ``model_json_schema()`` is ``schema`` itself."""
    from pydantic import BaseModel, ConfigDict

    class Args(BaseModel):
        model_config = ConfigDict(extra="allow")

        @classmethod
        def model_json_schema(cls, *args, **kwargs) -> dict:
            return {"type": "object", "properties": {}, **schema}

    Args.__name__ = Args.__qualname__ = f"{name}_args"
    return Args


def _function_tool(name: str, description: str, schema: dict, fn) -> Any:
    core = require("llama_index.core.tools", "llamaindex")

    def call(**kw):
        return to_text(fn(**kw))

    async def acall(**kw):
        return await arun(call, **kw)

    meta = core.ToolMetadata(name=name, description=description, fn_schema=schema_model(name, schema))
    return core.FunctionTool(fn=call, async_fn=acall, metadata=meta)


def tool_to_llamaindex(tool: "Tool", registry: "ToolRegistry | None" = None) -> Any:
    """One ``Tool`` as a LlamaIndex ``FunctionTool`` that runs via ``registry.call`` (or the tool itself)."""
    return _function_tool(tool.name, tool.description, tool.parameters, runner(tool, registry))


def to_llamaindex_tools(registry: "ToolRegistry", query: str, k: int = 20, **search_kw) -> list[Any]:
    """The ``k`` tools retrieved for ``query``, best match first, as LlamaIndex tools."""
    return [tool_to_llamaindex(t, registry) for t in retrieve(registry, query, k, **search_kw)]


def llamaindex_tools_by_name(registry: "ToolRegistry", names: Sequence[str]) -> list[Any]:
    """LlamaIndex tools for names the agent discovered through ``search_tools``."""
    return [tool_to_llamaindex(t, registry) for t in by_names(registry, names)]


def search_tools_llamaindex(registry: "ToolRegistry") -> Any:
    """A native ``search_tools`` tool returning matching tool definitions as JSON."""
    return _function_tool(registry.SEARCH_TOOL_NAME, SEARCH_TOOL_DESCRIPTION, SEARCH_TOOL_SCHEMA,
                          lambda query, k=5: search_results(registry, query, k))
