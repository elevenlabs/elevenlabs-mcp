"""
ElevenLabs MCP Server - Shared MCP and Client instances

This module contains the shared FastMCP server instance and ElevenLabs client
that are used across all tool modules.
"""

import os
import httpx

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs_mcp.utils import parse_location
from elevenlabs_mcp import __version__

from mcp.server.fastmcp import FastMCP

load_dotenv()

api_key = os.getenv("ELEVENLABS_API_KEY")
base_path = os.getenv("ELEVENLABS_MCP_BASE_PATH")
output_mode = os.getenv("ELEVENLABS_MCP_OUTPUT_MODE", "files").strip().lower()
DEFAULT_VOICE_ID = os.getenv("ELEVENLABS_DEFAULT_VOICE_ID", "cgSgspJ2msm6clMCkdW9")

if output_mode not in {"files", "resources", "both"}:
    raise ValueError("ELEVENLABS_MCP_OUTPUT_MODE must be one of: 'files', 'resources', 'both'")
if not api_key:
    raise ValueError("ELEVENLABS_API_KEY environment variable is required")

origin = parse_location(os.getenv("ELEVENLABS_API_RESIDENCY"))

custom_client = httpx.Client(
    headers={
        "User-Agent": f"ElevenLabs-MCP/{__version__}",
    },
)

client = ElevenLabs(api_key=api_key, httpx_client=custom_client, base_url=origin)
mcp = FastMCP("ElevenLabs")
