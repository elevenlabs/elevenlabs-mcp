# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an ElevenLabs MCP (Model Context Protocol) server that provides access to ElevenLabs' text-to-speech, speech-to-text, voice cloning, and conversational AI capabilities through MCP tools.

## Development Commands

```bash
# Setup development environment
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Run tests with coverage
./scripts/test.sh
./scripts/test.sh --verbose --fail-fast  # For quick feedback during development

# Run development server with MCP Inspector
./scripts/dev.sh
# Or directly: mcp dev elevenlabs_mcp/server.py

# Build package
./scripts/build.sh

# Deploy to PyPI (requires PyPI credentials)
./scripts/deploy.sh
```

## Architecture

### Core Components

1. **`server.py`** - Main MCP server implementation
   - Contains all 24 MCP tools decorated with `@mcp.tool`
   - Each tool that makes API calls includes cost warnings
   - Tools return `TextContent` with operation results

2. **`utils.py`** - Shared utilities
   - `make_output_path()` - Handles base path configuration
   - `make_output_file()` - Generates timestamped output filenames
   - `handle_input_file()` - Validates and resolves input file paths
   - `find_similar_files()` - Fuzzy file matching for better UX

3. **`convai.py`** - Conversational AI configuration builders
   - `create_conversation_config()` - Builds agent conversation settings
   - `create_platform_settings()` - Configures privacy and limits

4. **`model.py`** - Pydantic models for type safety

### Key Design Patterns

**Cost-Aware API Tools**: Every tool that calls ElevenLabs API has a cost warning in its description:
```python
@mcp.tool(
    description="""...
      COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.
    """
)
```

**File Path Handling**: All file operations respect the `ELEVENLABS_MCP_BASE_PATH` environment variable:
- If set: Files saved to specified directory
- If not set: Files saved to user's Desktop
- Input files can be absolute or relative paths

**Error Handling**: Custom `ElevenLabsMcpError` exception with helpful messages:
- File not found ’ Suggests similar files if available
- Permission errors ’ Clear guidance on file access issues

### Environment Configuration

Required environment variables:
- `ELEVENLABS_API_KEY` - Your ElevenLabs API key (required)
- `ELEVENLABS_MCP_BASE_PATH` - Base directory for file operations (optional)

### Adding New Tools

1. Add tool function in `server.py` with `@mcp.tool` decorator
2. Include cost warning if it makes API calls
3. Use consistent parameter patterns (see existing tools)
4. Return `TextContent` with clear success/error messages
5. Handle file operations through utility functions

### Testing

- Unit tests focus on utilities and file operations
- No integration tests for API calls (to avoid costs)
- Run tests before committing: `./scripts/test.sh`
- Aim for high coverage on utility functions

### Common Development Tasks

**Adding a new conversational AI feature**:
1. Check if ElevenLabs SDK supports it
2. Add/update configuration in `convai.py` if needed
3. Create tool in `server.py` following existing patterns
4. Test with dev server: `./scripts/dev.sh`

**Debugging file operations**:
- Set `ELEVENLABS_MCP_BASE_PATH` to a test directory
- Check file permissions with `handle_input_file()`
- Use `make_output_file()` for consistent naming

**Updating agent configurations**:
- Agent configs are immutable (Pydantic frozen models)
- Create new configs rather than modifying existing ones
- Use `create_conversation_config()` for proper structure