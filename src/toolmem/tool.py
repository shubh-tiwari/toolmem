"""Tool definitions: build from Python functions, OpenAI, Anthropic, or MCP schemas."""
from __future__ import annotations

import hashlib
import inspect
import json
import re
import types
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

_PY_TO_JSON = {str: "string", int: "integer", float: "number", bool: "boolean",
               list: "array", tuple: "array", set: "array", dict: "object"}


def _annotation_to_schema(ann: Any) -> dict:
    if ann is inspect.Parameter.empty or ann is Any:
        return {}
    origin, args = typing.get_origin(ann), typing.get_args(ann)
    if origin is typing.Union or origin is types.UnionType:
        non_none = [a for a in args if a is not type(None)]
        return _annotation_to_schema(non_none[0]) if len(non_none) == 1 else {}
    if origin is typing.Literal:
        return {"enum": list(args)}
    if origin in (list, tuple, set):
        schema: dict = {"type": "array"}
        if args and args[0] is not Ellipsis:
            item = _annotation_to_schema(args[0])
            if item:
                schema["items"] = item
        return schema
    if origin is dict:
        return {"type": "object"}
    if ann in _PY_TO_JSON:
        return {"type": _PY_TO_JSON[ann]}
    return {}


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into (summary, {param: description})."""
    if not doc:
        return "", {}
    doc = inspect.cleandoc(doc)
    parts = re.split(r"^\s*(?:Args|Arguments|Parameters)\s*:\s*$", doc, maxsplit=1, flags=re.M)
    summary = parts[0].strip()
    params: dict[str, str] = {}
    if len(parts) == 2:
        current = None
        for line in parts[1].splitlines():
            if re.match(r"^\s*(Returns|Raises|Yields|Examples?)\s*:", line):
                break
            m = re.match(r"^\s*(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$", line)
            if m:
                current = m.group(1)
                params[current] = m.group(2).strip()
            elif current and line.strip():
                params[current] += " " + line.strip()
    return summary, params


def _humanize(name: str) -> str:
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    return re.sub(r"[_\-.]+", " ", name).strip().lower()


@dataclass
class Tool:
    """A tool the agent can call.

    ``description`` is what the LLM sees in the tool schema. ``augmented_description``
    is only used for retrieval, so augmenting never changes what the model is told.
    """

    name: str
    description: str = ""
    parameters: dict = field(default_factory=lambda: {"type": "object", "properties": {}})
    func: Optional[Callable[..., Any]] = field(default=None, repr=False, compare=False)
    tags: list[str] = field(default_factory=list)
    augmented_description: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    # ---------- constructors ----------
    @classmethod
    def from_function(cls, func: Callable, *, name: str | None = None,
                      description: str | None = None, tags: list[str] | None = None) -> "Tool":
        summary, param_docs = _parse_docstring(func.__doc__ or "")
        try:
            hints = typing.get_type_hints(func)
        except Exception:
            hints = {}
        props: dict[str, dict] = {}
        required: list[str] = []
        for pname, p in inspect.signature(func).parameters.items():
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) or pname in ("self", "cls"):
                continue
            schema = _annotation_to_schema(hints.get(pname, p.annotation))
            if pname in param_docs:
                schema["description"] = param_docs[pname]
            if p.default is inspect.Parameter.empty:
                required.append(pname)
            elif isinstance(p.default, (str, int, float, bool)) or p.default is None:
                schema["default"] = p.default
            props[pname] = schema
        params: dict = {"type": "object", "properties": props}
        if required:
            params["required"] = required
        return cls(name=name or func.__name__, description=description or summary,
                   parameters=params, func=func, tags=list(tags or []))

    @classmethod
    def from_openai(cls, schema: dict, func: Callable | None = None, **kw) -> "Tool":
        fn = schema.get("function", schema)
        return cls(name=fn["name"], description=fn.get("description", ""),
                   parameters=fn.get("parameters") or {"type": "object", "properties": {}},
                   func=func, **kw)

    @classmethod
    def from_anthropic(cls, schema: dict, func: Callable | None = None, **kw) -> "Tool":
        return cls(name=schema["name"], description=schema.get("description", ""),
                   parameters=schema.get("input_schema") or {"type": "object", "properties": {}},
                   func=func, **kw)

    @classmethod
    def from_mcp(cls, schema: Any, func: Callable | None = None, **kw) -> "Tool":
        """Accepts an MCP tool as a dict or an object with name/description/inputSchema."""
        if not isinstance(schema, dict):
            schema = {"name": schema.name, "description": getattr(schema, "description", "") or "",
                      "inputSchema": getattr(schema, "inputSchema", None)}
        return cls(name=schema["name"], description=schema.get("description") or "",
                   parameters=schema.get("inputSchema") or {"type": "object", "properties": {}},
                   func=func, **kw)

    # ---------- exporters ----------
    def to_openai(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters}}

    def to_anthropic(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.parameters}

    def to_mcp(self) -> dict:
        return {"name": self.name, "description": self.description, "inputSchema": self.parameters}

    # ---------- retrieval ----------
    def base_text(self) -> str:
        """Original tool content as text (used as augmenter input and cache key)."""
        lines = [f"Tool: {self.name} ({_humanize(self.name)})"]
        if self.description:
            lines.append(self.description)
        for pname, p in (self.parameters.get("properties") or {}).items():
            desc = p.get("description", "")
            lines.append(f"- {pname}: {desc}".rstrip(": "))
        if self.tags:
            lines.append("Tags: " + ", ".join(self.tags))
        return "\n".join(lines)

    def embedding_text(self, use_augmented: bool = True) -> str:
        if use_augmented and self.augmented_description:
            return f"{self.base_text()}\n{self.augmented_description}"
        return self.base_text()

    def content_hash(self) -> str:
        payload = json.dumps([self.name, self.description, self.parameters, self.tags], sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def __call__(self, **kwargs: Any) -> Any:
        if self.func is None:
            raise RuntimeError(f"Tool '{self.name}' has no Python function attached.")
        return self.func(**kwargs)
