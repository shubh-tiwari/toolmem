"""Second tiny stdio MCP server for the proxy tests. Run as ``python tests/mcp_notes_server.py``."""
from mcp.server.mcpserver import MCPServer

server = MCPServer("notes")


@server.tool(description="Save a short text note under a title.")
def create_note(title: str, body: str) -> str:
    return f"saved {title}"


@server.tool(description="Find saved notes whose text mentions a keyword.")
def find_notes(keyword: str) -> str:
    return f"no notes mention {keyword}"


@server.tool(description="Always fails; used to check that upstream errors pass through.")
def broken() -> str:
    raise ValueError("this tool always fails")


if __name__ == "__main__":
    server.run("stdio")
