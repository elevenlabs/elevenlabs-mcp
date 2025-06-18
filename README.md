![export](https://github.com/user-attachments/assets/ee379feb-348d-48e7-899c-134f7f7cd74f)

<div class="title-block" style="text-align: center;" align="center">

  [![Discord Community](https://img.shields.io/badge/discord-@elevenlabs-000000.svg?style=for-the-badge&logo=discord&labelColor=000)](https://discord.gg/elevenlabs)
  [![Twitter](https://img.shields.io/badge/Twitter-@elevenlabsio-000000.svg?style=for-the-badge&logo=twitter&labelColor=000)](https://x.com/ElevenLabsDevs)
  [![PyPI](https://img.shields.io/badge/PyPI-elevenlabs--mcp-000000.svg?style=for-the-badge&logo=pypi&labelColor=000)](https://pypi.org/project/elevenlabs-mcp)
  [![Tests](https://img.shields.io/badge/tests-passing-000000.svg?style=for-the-badge&logo=github&labelColor=000)](https://github.com/elevenlabs/elevenlabs-mcp-server/actions/workflows/test.yml)

</div>


<p align="center">
  Official ElevenLabs <a href="https://github.com/modelcontextprotocol">Model Context Protocol (MCP)</a> server that enables interaction with powerful Text to Speech and audio processing APIs. This server allows MCP clients like <a href="https://www.anthropic.com/claude">Claude Desktop</a>, <a href="https://www.cursor.so">Cursor</a>, <a href="https://codeium.com/windsurf">Windsurf</a>, <a href="https://github.com/openai/openai-agents-python">OpenAI Agents</a> and others to generate speech, clone voices, transcribe audio, and more.
</p>

## Enhanced Features

This fork adds **configurable TTS model selection** to the official ElevenLabs MCP server:

- ✨ **Model Selection**: Choose any ElevenLabs TTS model via parameter or environment variable
- 🔧 **Environment Configuration**: Set `ELEVENLABS_MODEL_ID` for deployment flexibility  
- 🔄 **Backward Compatible**: Maintains all existing functionality without breaking changes
- 📝 **Simple Enhancement**: Only 6 lines of code for significant functionality improvement

### New Model Selection Options

The `text_to_speech` function now supports these models via the `model_id` parameter:
- `eleven_v3` - Most expressive model with audio tags support (70+ languages)
- `eleven_multilingual_v2` - High quality multilingual model (29 languages) 
- `eleven_flash_v2_5` - Fastest model with ultra-low latency (32 languages)
- `eleven_turbo_v2_5` - Balanced quality and speed (32 languages)
- `eleven_flash_v2` - Fast English-only model
- `eleven_turbo_v2` - Balanced English-only model
- `eleven_monolingual_v1` - Legacy English model

### Configuration Options

1. **Function Parameter**: Pass `model_id` directly to `text_to_speech`
2. **Environment Variable**: Set `ELEVENLABS_MODEL_ID` in your environment
3. **Fallback**: Uses original language-based selection as fallback

## Quickstart with Claude Desktop

1. Get your API key from [ElevenLabs](https://elevenlabs.io/app/settings/api-keys). There is a free tier with 10k credits per month.
2. Install `uv` (Python package manager), install with `curl -LsSf https://astral.sh/uv/install.sh | sh` or see the `uv` [repo](https://github.com/astral-sh/uv) for additional install methods.
3. Go to Claude > Settings > Developer > Edit Config > claude_desktop_config.json to include the following:

```json
{
  "mcpServers": {
    "ElevenLabs": {
      "command": "uvx",
      "args": ["elevenlabs-mcp"],
      "env": {
        "ELEVENLABS_API_KEY": "<insert-your-api-key-here>",
        "ELEVENLABS_MODEL_ID": "eleven_v3"
      }
    }
  }
}
```

If you're using Windows, you will have to enable "Developer Mode" in Claude Desktop to use the MCP server. Click "Help" in the hamburger menu at the top left and select "Enable Developer Mode".

## Other MCP clients

For other clients like Cursor and Windsurf, run:
1. `pip install elevenlabs-mcp`
2. `python -m elevenlabs_mcp --api-key={{PUT_YOUR_API_KEY_HERE}} --print` to get the configuration. Paste it into appropriate configuration directory specified by your MCP client.

That's it. Your MCP client can now interact with ElevenLabs through these tools:

## Example usage

⚠️ Warning: ElevenLabs credits are needed to use these tools.

Try asking Claude:

- "Create an AI agent that speaks like a film noir detective and can answer questions about classic movies"
- "Generate three voice variations for a wise, ancient dragon character, then I will choose my favorite voice to add to my voice library"
- "Convert this recording of my voice to sound like a medieval knight"
- "Create a soundscape of a thunderstorm in a dense jungle with animals reacting to the weather"
- "Turn this speech into text, identify different speakers, then convert it back using unique voices for each person"
- "Use the eleven_v3 model for maximum expressiveness in this TTS generation"

## Optional features

You can add the `ELEVENLABS_MCP_BASE_PATH` environment variable to the `claude_desktop_config.json` to specify the base path MCP server should look for and output files specified with relative paths.

## Contributing

If you want to contribute or run from source:

1. Clone the repository:

```bash
git clone https://github.com/elevenlabs/elevenlabs-mcp
cd elevenlabs-mcp
```

2. Create a virtual environment and install dependencies [using uv](https://github.com/astral-sh/uv):

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

3. Copy `.env.example` to `.env` and add your ElevenLabs API key:

```bash
cp .env.example .env
# Edit .env and add your API key
```

4. Run the tests to make sure everything is working:

```bash
./scripts/test.sh
# Or with options
./scripts/test.sh --verbose --fail-fast
```

5. Install the server in Claude Desktop: `mcp install elevenlabs_mcp/server.py`

6. Debug and test locally with MCP Inspector: `mcp dev elevenlabs_mcp/server.py`

## Troubleshooting

Logs when running with Claude Desktop can be found at:

- **Windows**: `%APPDATA%\Claude\logs\mcp-server-elevenlabs.log`
- **macOS**: `~/Library/Logs/Claude/mcp-server-elevenlabs.log`

### Timeouts when using certain tools

Certain ElevenLabs API operations, like voice design and audio isolation, can take a long time to resolve. When using the MCP inspector in dev mode, you might get timeout errors despite the tool completing its intended task.

This shouldn't occur when using a client like Claude.

### MCP ElevenLabs: spawn uvx ENOENT

If you encounter the error "MCP ElevenLabs: spawn uvx ENOENT", confirm its absolute path by running this command in your terminal:

```bash
which uvx
```

Once you obtain the absolute path (e.g., `/usr/local/bin/uvx`), update your configuration to use that path (e.g., `"command": "/usr/local/bin/uvx"`). This ensures that the correct executable is referenced.
