"""OpenAI Agents SDK adapter: registry tools as ``FunctionTool`` objects (``pip install 'toolmem[openai-agents]'``).

    agent = Agent(name="assistant", tools=to_openai_agents_tools(reg, user_message, k=20))

    agent = Agent(name="assistant", tools=[search_tools_openai_agents(reg)])   # or let it search
    # between runs, add what it found:
    agent = with_tools(agent, reg, [d["name"] for d in json.loads(search_output)])

Schemas are passed through with ``strict_json_schema=False``: most published tool schemas (MCP
especially) do not meet OpenAI's strict-mode rules.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Sequence

from . import (SEARCH_TOOL_DESCRIPTION, SEARCH_TOOL_SCHEMA, arun, by_names, require, retrieve, runner,
               search_results, to_text)

if TYPE_CHECKING:
    from ..registry import ToolRegistry
    from ..tool import Tool


def _function_tool(name: str, description: str, schema: dict, fn) -> Any:
    agents = require("agents", "openai-agents")

    async def invoke(ctx: Any, args: str) -> str:
        return to_text(await arun(fn, **json.loads(args or "{}")))

    return agents.FunctionTool(name=name, description=description, params_json_schema=schema,
                               on_invoke_tool=invoke, strict_json_schema=False)


def tool_to_openai_agents(tool: "Tool", registry: "ToolRegistry | None" = None) -> Any:
    """One ``Tool`` as an Agents SDK ``FunctionTool`` that runs via ``registry.call`` (or the tool itself)."""
    return _function_tool(tool.name, tool.description, tool.parameters, runner(tool, registry))


def to_openai_agents_tools(registry: "ToolRegistry", query: str, k: int = 20, **search_kw) -> list[Any]:
    """The ``k`` tools retrieved for ``query``, best match first, as Agents SDK tools."""
    return [tool_to_openai_agents(t, registry) for t in retrieve(registry, query, k, **search_kw)]


def openai_agents_tools_by_name(registry: "ToolRegistry", names: Sequence[str]) -> list[Any]:
    """Agents SDK tools for names the agent discovered through ``search_tools``."""
    return [tool_to_openai_agents(t, registry) for t in by_names(registry, names)]


def search_tools_openai_agents(registry: "ToolRegistry") -> Any:
    """A native ``search_tools`` tool returning matching tool definitions as JSON."""
    return _function_tool(registry.SEARCH_TOOL_NAME, SEARCH_TOOL_DESCRIPTION, SEARCH_TOOL_SCHEMA,
                          lambda query, k=5: search_results(registry, query, k))


def with_tools(agent: Any, registry: "ToolRegistry", names: Sequence[str]) -> Any:
    """A copy of ``agent`` that also has the named registry tools (ones it already has are kept once)."""
    have = {t.name for t in agent.tools}
    new = [t for t in openai_agents_tools_by_name(registry, names) if t.name not in have]
    return agent.clone(tools=[*agent.tools, *new])
