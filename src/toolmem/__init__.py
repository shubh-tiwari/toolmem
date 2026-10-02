"""toolmem: semantic tool memory for LLM agents."""
from .augment import CallableAugmenter, LLMAugmenter
from .backends import ChromaBackend, InMemoryBackend, LanceDBBackend, PgVectorBackend, VectorBackend
from .cache import SQLiteCache
from .casegen import CheckReport, LLMCaseGenerator, check_cases, generate_cases
from .embedders import CallableEmbedder, HashingEmbedder, OpenAIEmbedder, SentenceTransformerEmbedder
from .evals import EvalCase, EvalReport, SelectionReport, calibration, compare, evaluate, evaluate_selection, load_cases
from .mcp_loader import MCPServerConfig, add_mcp_server, load_mcp_tools, load_mcp_tools_async, load_server_configs
from .registry import SearchResult, ToolRegistry
from .rerank import CallableReranker, Decision, JevReranker, Reranker
from .tool import Tool

__version__ = "0.1.0"
__all__ = ["Tool", "ToolRegistry", "SearchResult", "HashingEmbedder", "SentenceTransformerEmbedder",
           "OpenAIEmbedder", "CallableEmbedder", "InMemoryBackend", "VectorBackend", "LLMAugmenter",
           "CallableAugmenter", "SQLiteCache", "EvalCase", "EvalReport", "evaluate", "compare", "load_cases",
           "Reranker", "JevReranker", "CallableReranker", "Decision", "SelectionReport", "evaluate_selection",
           "calibration", "ChromaBackend", "LanceDBBackend", "PgVectorBackend", "MCPServerConfig",
           "load_mcp_tools", "load_mcp_tools_async", "add_mcp_server", "generate_cases",
           "check_cases", "CheckReport", "LLMCaseGenerator", "load_server_configs"]
