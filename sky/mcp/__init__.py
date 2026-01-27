"""SkyPilot MCP (Model Context Protocol) server.

This module provides an MCP server that exposes SkyPilot functionality
to AI assistants and other MCP clients.
"""

from sky.mcp.server import main
from sky.mcp.server import run_server

__all__ = ['main', 'run_server']
