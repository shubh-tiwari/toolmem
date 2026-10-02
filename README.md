# toolmem

**Semantic tool memory for LLM agents.** Register hundreds of tools, send the model only the few each query needs, and measure whether retrieval actually picks the right ones.

Putting every tool definition in the prompt bloats context, raises cost and latency, and makes tool selection worse as the catalog grows. `toolmem` treats tools as retrievable memory: tools are indexed once, and at inference time a hybrid (embedding + BM25) search returns the top-k relevant schemas.

## Benchmark

With 454 real tools from 34 MCP servers in the prompt, two low-cost LLMs pick an acceptable tool 87
to 88% of the time, at about 144,000 input tokens per request. With 10 tools they pick correctly 99%
of the time. A toolmem shortlist of 20 gets the same accuracy (90 to 93%, within noise of sending
everything) on about 1/18 of the tokens. DeepSeek leans on toolmem listing the best match first
(84% when the shortlist is shuffled); Qwen and Jev do not. Tested on 97 queries.

![Accuracy vs. tools shown](https://raw.githubusercontent.com/shubh-tiwari/toolmem/main/benchmark/results/accuracy_vs_tools.png)

Picking from the shortlist with TypeSafe's Jev instead of an LLM scores 94% in 0.5 seconds on about
1,700 tokens, at $0.07 per thousand requests, against 8.6 seconds and $2.46 billed ($11.31 at list
price) for DeepSeek V4 Flash with every tool in the prompt. Jev chooses the tool; an LLM still has
to write its arguments. Jev's picks at 0.9 confidence or higher, 58% of queries, were all correct. Method, full results, caveats and how to re-run: [benchmark/](https://github.com/shubh-tiwari/toolmem/blob/main/benchmark/README.md).

## Features

- **Register tools from anywhere:** Python functions (schema built from type hints and docstrings), OpenAI, Anthropic, or MCP tool definitions, or connect to MCP servers and index their tools directly.
- **Hybrid retrieval:** semantic search fused with BM25 keyword search via reciprocal rank fusion, so both paraphrased requests and exact tool names match.
- **LLM augmentation:** optionally rewrite each tool's description into retrieval-friendly text (example requests, when to use it and when not). Works with any OpenAI-compatible endpoint, including OpenRouter. The augmented text is used **only for retrieval**; the schema sent to the model keeps your original description.
- **Pick one tool with a confidence:** `select()` retrieves a shortlist and lets a pluggable reranker choose one tool, with a probability. `JevReranker` uses TypeSafe's Jev.
- **Built-in evals:** a `toolmem eval` command, LLM-drafted test queries with leakage checks, hit@k, recall@k, and MRR, with side-by-side comparisons of modes, embedders, and augmentation, plus end-to-end selection accuracy and calibration (are 90%-confidence picks right 90% of the time?).
- **Caching:** embeddings and augmentations are stored in SQLite and recomputed only when a tool changes.
- **Toolbox meta-tool:** give the agent a `search_tools` tool so it can discover tools on demand.
- **MCP proxy:** `toolmem proxy` puts one MCP server in front of all of yours and shows the client only the tools each request needs.
- **Framework adapters:** retrieved tools as native LangChain, LlamaIndex or OpenAI Agents SDK tools.
- **Vector stores:** exact in-memory search by default, or Chroma, LanceDB or pgvector.
- **Tiny core:** only depends on `numpy`. Embedding and LLM providers, vector stores and MCP are optional extras.

## Install

```bash
git clone https://github.com/shubh-tiwari/toolmem.git
cd toolmem
pip install -e .                 # core (numpy only)
pip install -e ".[openai,mcp]"   # optional extras, for example OpenAI embeddings and MCP
```

Optional extras: `local` (sentence-transformers), `openai` (OpenAI-compatible embeddings and
augmentation), `mcp` (load tools from MCP servers, and the proxy), `chroma`, `lancedb`, `pgvector`
(vector stores), `langchain`, `llamaindex`, `openai-agents` (framework adapters).

## Quickstart

```python
from toolmem import ToolRegistry, SentenceTransformerEmbedder

reg = ToolRegistry(embedder=SentenceTransformerEmbedder("all-MiniLM-L6-v2"))

@reg.register(tags=["weather"])
def get_weather(city: str, days: int = 1) -> str:
    """Get the weather forecast for a city.

    Args:
        city: City name, e.g. "London".
        days: Number of days to forecast.
    """
    ...

reg.add_many([...])  # or Tool.from_openai(...), Tool.from_mcp(...)

# Retrieve only the relevant schemas and pass them to your LLM call
tools = reg.tools_for("will it rain in London this weekend", k=20, format="openai")

# Execute the tool call the model returns
result = reg.call(tool_call.function.name, tool_call.function.arguments)
```

Use `format="anthropic"` or `format="mcp"` for other APIs.

**How many tools to retrieve.** On the benchmark's 454-tool catalog, a top-5 shortlist left out the
right tool for 15% of queries and scored no better than sending every tool. A top-20 shortlist
contained it 98% of the time and matched full-catalog accuracy on about 1/18 of the tokens, so `k=20`
is a better starting point than the default of 5 for large catalogs. Those numbers used OpenAI's
`text-embedding-3-small` (`OpenAIEmbedder`) with rewritten descriptions (see below); a small local
model such as `all-MiniLM-L6-v2` is free and private but may retrieve less accurately, so run the
evals on your own catalog.

## LLM augmentation (e.g. via OpenRouter)

```python
import os
from toolmem import ToolRegistry, LLMAugmenter, SQLiteCache

cache = SQLiteCache(".toolmem_cache.sqlite")
aug = LLMAugmenter(
    model="deepseek/deepseek-v4.1-flash",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    extra_body={"reasoning": {"enabled": False}},  # reasoning models can otherwise return empty text
    cache=cache,
)
reg = ToolRegistry(embedder=..., augmenter=aug, cache=cache)
```

Each tool is augmented once per version. A cheap model is usually enough, since the output is search-index text, not something users read. On the benchmark, rewritten descriptions raised the share of queries whose right tool made the top 5 from 72% to 85%, and top-5 accuracy by 15 points.

## Evaluate retrieval

```python
from toolmem import EvalCase, evaluate, compare

cases = [
    EvalCase("will it rain in London this weekend", ["get_weather_forecast"]),
    EvalCase("how many rupees is 50 dollars", ["convert_currency"]),
]
reports = [evaluate(reg, cases, mode=m, label=m) for m in ("keyword", "semantic", "hybrid")]
print(compare(reports))
for f in reports[-1].failures:
    print(f)
```

Cases can also be loaded from JSONL with `load_cases("cases.jsonl")`, one `{"query": ..., "expected": [...]}` per line.

Tips for honest evals: write queries the way users phrase them (not copied from tool descriptions), include confusable pairs (current weather vs. forecast), and don't generate eval queries with the same model and prompt you use for augmentation.

## Generate test queries, with leakage checks

Writing test queries by hand is slow. `generate_cases` drafts them with an LLM: for a sample of
tools spread across servers, it shows the model the tool and its closest siblings and asks for
requests phrased the way users talk, including one designed to be confused with a sibling.
`check_cases` then flags unknown tools, duplicates, unreviewed rows, and queries that reuse most of
the tool's own wording or its augmented description.

```python
import os
from toolmem import LLMCaseGenerator, check_cases, generate_cases

gen = LLMCaseGenerator(model="anthropic/claude-opus-5.5", base_url="https://openrouter.ai/api/v1",
                       api_key=os.environ["OPENROUTER_API_KEY"])
drafts = generate_cases(list(reg), gen, n_tools=40, cache=cache)   # rows with status "draft"

# Review each row: set "status" to "ok" or "dropped", fix wording, add equally valid tools.
report = check_cases(reviewed, list(reg), generator_model=gen.model, augmenter_model=aug.model)
print(report)          # errors fail the check; warnings flag likely leakage
```

`generate_cases` also accepts any `prompt -> str` function, for example a wrapper around the
Anthropic SDK. Passing `generator_model` and `augmenter_model` warns when both come from the same
model family, since augmented descriptions from that family may match its own queries too well.
Drafts still need a human review: the model cannot tell when another tool in the catalog is equally
correct.

## Evaluate from the command line

`toolmem eval` runs the same steps on your own catalog without writing code. The catalog comes from
a tools file (`ToolRegistry.save` format) or straight from an MCP config in the
`{"mcpServers": {...}}` layout, such as a project's `.mcp.json` or Claude Desktop's config:

```bash
toolmem eval draft --mcp-config .mcp.json --model anthropic/claude-opus-5.5 --out cases.draft.jsonl --yes
# review cases.draft.jsonl: set "status" to "ok" or "dropped", save as cases.jsonl
toolmem eval check cases.jsonl --mcp-config .mcp.json
toolmem eval score cases.jsonl --mcp-config .mcp.json --k 1,5,20 --embedder openai --yes
```

`score` prints hit@k and MRR per retrieval mode and the queries it missed; `--json` saves the
results and `--select` also scores a reranker's picks. It defaults to the free, offline `hashing`
embedder. Any step that would call a paid API (drafting, OpenAI embeddings, rewriting descriptions,
Jev) prints how many calls it would make and stops unless you pass `--yes`; cached results are
reused and not counted.

## Let the agent search (toolbox pattern)

```python
tools = [reg.search_tool()]          # expose the meta-tool
# when the model calls "search_tools":
found = reg.handle_search_tool(tool_call.function.arguments)
tools += found                        # add discovered tools for the next turn
```

## Pick one tool with a reranker

Retrieval can hand a reranker a wide shortlist, and the reranker picks one tool with a confidence.
`JevReranker` uses TypeSafe's Jev, which answers a multiple-choice question over the candidates and
returns a probability for each.

```python
from toolmem import ToolRegistry, JevReranker

reg = ToolRegistry(embedder=..., reranker=JevReranker())   # TYPESAFE_API_KEY, or provider="cloudflare"
decision = reg.select("refund my last order", candidates=40)
decision.name, decision.confidence    # e.g. ("stripe__create_refund", 0.94)
if decision.confidence < 0.9:
    shortlist = decision.top(3)       # hand three options to an LLM, or ask the user
```

`evaluate_selection(reg, cases)` reports accuracy, calibration error, and how many requests each
confidence threshold covers, so you can check that 0.9-confidence picks are right about 90% of the
time before gating on them.

On the benchmark, Jev picking from a toolmem shortlist scored 94% with 20 candidates and 96% with
50, in about 0.5 seconds. Its picks at confidence 0.9 or higher were all correct. Handing the less
confident picks to an LLM with `decision.top(3)` did not improve accuracy there, so measure it on
your own catalog before adding that step. Jev accepts at most 254 candidates plus a `none` option.

## Load tools from MCP servers

```python
from toolmem import MCPServerConfig, ToolRegistry, add_mcp_server

reg = ToolRegistry(embedder=...)
github = MCPServerConfig(name="github", command="npx", args=["-y", "@modelcontextprotocol/server-github"],
                         env={"GITHUB_PERSONAL_ACCESS_TOKEN": "..."})
add_mcp_server(reg, github, call=True)      # tools are named github__create_issue, etc.
reg.search("open a bug report for the login crash", k=20)
reg.call("github__create_issue", {"owner": "acme", "repo": "app", "title": "Login crash"})
```

The server is started over stdio just long enough to list its tools; nothing runs until you call
one. Names get the server as a prefix, because several servers ship a tool called `search`
(`prefix=False` turns that off). With `call=True` each tool can be executed through
`reg.call`, which starts the server for that one call: simple, not fast. For a long-lived session,
keep your own MCP client and use `load_mcp_tools(server)` to get the definitions only. Use the
`*_async` variants inside a running event loop.

## MCP proxy

`toolmem proxy` is itself an MCP server. It starts every server in an MCP config once, indexes
their tools, and exposes two tools instead of hundreds: `search_tools` returns the definitions of
the tools that fit a request, and `call_tool` forwards a call to the right server.

```json
{"mcpServers": {"toolmem": {"command": "toolmem",
                            "args": ["proxy", "/path/to/upstream.json", "--k", "20"]}}}
```

`upstream.json` lists the servers to proxy in the same `{"mcpServers": {...}}` layout, so you can
move your existing entries into it. With `--mode dynamic` the proxy instead lists the tools from
the latest search directly and notifies the client that its tool list changed; use it with
clients that reload their tool list on that notification, and `--pin` tools that should always be
listed. Retrieval defaults to the offline `hashing` embedder; `--embedder openai --cache
.toolmem_cache.sqlite` retrieves better and embeds each tool once. Only stdio servers are
supported, and a server that fails to start is skipped with a message on stderr.

## Framework adapters

Each adapter turns retrieved tools into the framework's own tool objects, which run through
`reg.call`. Import them from `toolmem.adapters.<framework>`; importing `toolmem` alone never imports
a framework.

```python
from toolmem.adapters.langchain import to_langchain_tools, search_tools_langchain
from toolmem.adapters.llamaindex import to_llamaindex_tools
from toolmem.adapters.openai_agents import to_openai_agents_tools, with_tools

tools = to_langchain_tools(reg, user_message, k=20)        # per request: StructuredTool objects
tools = to_llamaindex_tools(reg, user_message, k=20)       # FunctionTool objects
tools = to_openai_agents_tools(reg, user_message, k=20)    # Agents SDK FunctionTool objects

finder = search_tools_langchain(reg)                       # or let the agent search for tools
agent = with_tools(agent, reg, ["github__create_issue"])   # Agents SDK: add discovered tools
```

Schemas are passed through unchanged (the Agents SDK tools use non-strict schemas, since MCP
schemas rarely meet strict mode's rules). The adapters were tested with langchain-core 1.6,
llama-index-core 0.14 and openai-agents 0.22, calling each tool through the framework's own API;
they have not yet been run inside a live agent loop.

## Vector store backends

The default backend is exact in-memory cosine search, which handles tens of thousands of tools. To
keep the index in a vector store:

```python
from toolmem import ChromaBackend, LanceDBBackend, PgVectorBackend, ToolRegistry

reg = ToolRegistry(embedder=..., backend=ChromaBackend(path=".chroma"))
reg = ToolRegistry(embedder=..., backend=LanceDBBackend(uri=".toolmem_lancedb"))
reg = ToolRegistry(embedder=..., backend=PgVectorBackend(dsn="postgresql://localhost/app"))
```

All three return cosine similarity, so results match the in-memory backend. `PgVectorBackend`
creates the `vector` extension and its table on first use and searches exactly; add an HNSW index
yourself for very large catalogs. To plug in another store, implement `upsert`, `delete`, `search`, `__contains__`,
and `__len__` (see `toolmem.backends.VectorBackend`) and pass it as `ToolRegistry(backend=...)`.

## Acknowledgements

The core idea of treating tools as procedural memory with LLM-enhanced descriptions is inspired by the DeepLearning.AI short course *Agent Memory: Building Memory-Aware Agents*. This is an independent implementation.

## License

MIT
