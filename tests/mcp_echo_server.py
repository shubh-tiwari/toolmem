"""Tiny stdio MCP server for the loader tests. Run as ``python tests/mcp_echo_server.py``."""
from mcp.server.mcpserver import MCPServer

server = MCPServer("echo")


@server.tool(description="Repeat a message back, optionally in upper case.")
def echo(message: str, shout: bool = False) -> str:
    return message.upper() if shout else message


@server.tool(name="add.numbers", description="Add two integers and return the sum.")
def add_numbers(a: int, b: int) -> str:
    return str(a + b)


@server.tool(description="Look up the current weather forecast for a city.")
def get_weather(city: str) -> str:
    return f"sunny in {city}"


if __name__ == "__main__":
    server.run("stdio")
