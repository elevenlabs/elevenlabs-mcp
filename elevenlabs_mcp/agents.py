"""
ElevenLabs MCP Server - Conversational AI Agents and Tools Management

This module contains tools for managing conversational AI agents and their
server tools (webhooks and client tools).
"""

from datetime import datetime
from typing import Literal
from io import BytesIO

from elevenlabs.types import (
    ToolRequestModel,
    ToolRequestModelToolConfig_Client,
    ToolRequestModelToolConfig_Webhook,
    WebhookToolApiSchemaConfigInput,
    ObjectJsonSchemaPropertyInput,
    LiteralJsonSchemaProperty,
    QueryParamsJsonSchema,
)
from elevenlabs.types.knowledge_base_locator import KnowledgeBaseLocator
from elevenlabs_mcp.convai import create_conversation_config, create_platform_settings
from elevenlabs_mcp.mcp import mcp, client, DEFAULT_VOICE_ID
from elevenlabs_mcp.utils import Schemas, make_error, handle_input_file
from mcp.types import TextContent


@mcp.tool(
    description="""Create a conversational AI agent with custom configuration.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.

    Args:
        name: Name of the agent
        first_message: First message the agent will say i.e. "Hi, how can I help you today?"
        system_prompt: System prompt for the agent
        voice_id: ID of the voice to use for the agent
        language: ISO 639-1 language code for the agent
        llm: LLM to use for the agent
        temperature: Temperature for the agent. The lower the temperature, the more deterministic the agent's responses will be. Range is 0 to 1.
        max_tokens: Maximum number of tokens to generate.
        asr_quality: Quality of the ASR. `high` or `low`.
        model_id: ID of the ElevenLabs model to use for the agent.
        optimize_streaming_latency: Optimize streaming latency. Range is 0 to 4.
        stability: Stability for the agent. Range is 0 to 1.
        similarity_boost: Similarity boost for the agent. Range is 0 to 1.
        turn_timeout: Timeout for the agent to respond in seconds. Defaults to 7 seconds.
        max_duration_seconds: Maximum duration of a conversation in seconds. Defaults to 600 seconds (10 minutes).
        record_voice: Whether to record the agent's voice.
        retention_days: Number of days to retain the agent's data.
    """
)
def create_agent(
    name: str,
    first_message: str,
    system_prompt: str,
    voice_id: str | None = DEFAULT_VOICE_ID,
    language: str = "en",
    llm: str = "gemini-2.0-flash-001",
    temperature: float = 0.5,
    max_tokens: int | None = None,
    asr_quality: str = "high",
    model_id: str = "eleven_turbo_v2",
    optimize_streaming_latency: int = 3,
    stability: float = 0.5,
    similarity_boost: float = 0.8,
    turn_timeout: int = 7,
    max_duration_seconds: int = 300,
    record_voice: bool = True,
    retention_days: int = 730,
) -> TextContent:
    conversation_config = create_conversation_config(
        language=language,
        system_prompt=system_prompt,
        llm=llm,
        first_message=first_message,
        temperature=temperature,
        max_tokens=max_tokens,
        asr_quality=asr_quality,
        voice_id=voice_id,
        model_id=model_id,
        optimize_streaming_latency=optimize_streaming_latency,
        stability=stability,
        similarity_boost=similarity_boost,
        turn_timeout=turn_timeout,
        max_duration_seconds=max_duration_seconds,
    )

    platform_settings = create_platform_settings(
        record_voice=record_voice,
        retention_days=retention_days,
    )

    response = client.conversational_ai.agents.create(
        name=name,
        conversation_config=conversation_config,
        platform_settings=platform_settings,
    )

    return TextContent(
        type="text",
        text=f"""Agent created successfully: Name: {name}, Agent ID: {response.agent_id}, System Prompt: {system_prompt}, Voice ID: {voice_id or "Default"}, Language: {language}, LLM: {llm}, You can use this agent ID for future interactions with the agent.""",
    )


@mcp.tool(
    description="""Add a knowledge base to ElevenLabs workspace. Allowed types are epub, pdf, docx, txt, html.

    ⚠️ COST WARNING: This tool makes an API call to ElevenLabs which may incur costs. Only use when explicitly requested by the user.

    Args:
        agent_id: ID of the agent to add the knowledge base to.
        knowledge_base_name: Name of the knowledge base.
        url: URL of the knowledge base.
        input_file_path: Path to the file to add to the knowledge base.
        text: Text to add to the knowledge base.
    """
)
def add_knowledge_base_to_agent(
    agent_id: str,
    knowledge_base_name: str,
    url: str | None = None,
    input_file_path: str | None = None,
    text: str | None = None,
) -> TextContent:
    provided_params = [
        param for param in [url, input_file_path, text] if param is not None
    ]
    if len(provided_params) == 0:
        make_error("Must provide either a URL, a file, or text")
    if len(provided_params) > 1:
        make_error("Must provide exactly one of: URL, file, or text")

    if url is not None:
        response = client.conversational_ai.knowledge_base.documents.create_from_url(
            name=knowledge_base_name,
            url=url,
        )
    else:
        if text is not None:
            text_bytes = text.encode("utf-8")
            text_io = BytesIO(text_bytes)
            text_io.name = "text.txt"
            text_io.content_type = "text/plain"
            file = text_io
        elif input_file_path is not None:
            path = handle_input_file(
                file_path=input_file_path, audio_content_check=False
            )
            file = open(path, "rb")

        response = client.conversational_ai.knowledge_base.documents.create_from_file(
            name=knowledge_base_name,
            file=file,
        )

    agent = client.conversational_ai.agents.get(agent_id=agent_id)

    agent_config = agent.conversation_config.agent
    knowledge_base_list = (
        agent_config.get("prompt", {}).get("knowledge_base", []) if agent_config else []
    )
    knowledge_base_list.append(
        KnowledgeBaseLocator(
            type="file" if file else "url",
            name=knowledge_base_name,
            id=response.id,
        )
    )

    if agent_config and "prompt" not in agent_config:
        agent_config["prompt"] = {}
    if agent_config:
        agent_config["prompt"]["knowledge_base"] = knowledge_base_list

    client.conversational_ai.agents.update(
        agent_id=agent_id, conversation_config=agent.conversation_config
    )
    return TextContent(
        type="text",
        text=f"""Knowledge base created with ID: {response.id} and added to agent {agent_id} successfully.""",
    )


@mcp.tool(description="List all available conversational AI agents")
def list_agents() -> TextContent:
    """List all available conversational AI agents.

    Returns:
        TextContent with a formatted list of available agents
    """
    response = client.conversational_ai.agents.list()

    if not response.agents:
        return TextContent(type="text", text="No agents found.")

    agents = ", ".join(
        f"{agent.name} (ID: {agent.agent_id})" for agent in response.agents
    )
    return TextContent(type="text", text=f"Available agents: {agents}")

@mcp.tool(description="Get details about a specific conversational AI agent")
def get_agent(agent_id: str) -> TextContent:
    """Get details about a specific conversational AI agent.

    Args:
        agent_id: The ID of the agent to retrieve

    Returns:
        TextContent with detailed information about the agent
    """
    response = client.conversational_ai.agents.get(agent_id=agent_id)

    voice_info = "None"
    if response.conversation_config.tts:
        voice_info = f"Voice ID: {response.conversation_config.tts.voice_id}"

    return TextContent(
        type="text",
        text=f"Agent Details: Name: {response.name}, Agent ID: {response.agent_id}, Voice Configuration: {voice_info}, Created At: {datetime.fromtimestamp(response.metadata.created_at_unix_secs).strftime('%Y-%m-%d %H:%M:%S')}",
    )


@mcp.tool(description="List all tools available in the workspace")
def list_workspace_tools() -> TextContent:
    """List all tools available in the ElevenLabs workspace.

    Returns:
        TextContent with a formatted list of available tools
    """
    response = client.conversational_ai.tools.list()

    if not response.tools:
        return TextContent(type="text", text="No tools found in workspace.")

    tools = []
    for tool in response.tools:
        tools.append(
            f"- {getattr(tool.tool_config, 'name', 'unnamed')}"
            f" (ID: {tool.id}, Type: {getattr(tool.tool_config, 'type', 'unknown')})"
        )
    txt = '\n'.join(tools)
    return TextContent(
        type="text",
        text=f"Workspace Tools ({len(response.tools)} total):\n{txt}",
    )


@mcp.tool(description="Get details about a specific tool")
def get_tool(tool_id: str) -> TextContent:
    """Get details about a specific tool in the workspace.

    Args:
        tool_id: The ID of the tool to retrieve

    Returns:
        TextContent with detailed information about the tool
    """
    response = client.conversational_ai.tools.get(tool_id=tool_id)

    config = response.tool_config
    tool_type = config.type if hasattr(config, "type") else "unknown"

    details = [
        f"Tool ID: {response.id}",
        f"Type: {tool_type}",
        f"Name: {config.name}",
    ]

    if hasattr(config, "description") and config.description:
        details.append(f"Description: {config.description}")

    if tool_type == "webhook" and hasattr(config, "api_schema"):
        schema = config.api_schema
        details.append(f"URL: {schema.url}")
        if hasattr(schema, "method") and schema.method:
            details.append(f"Method: {schema.method}")

    if hasattr(config, "response_timeout_secs") and config.response_timeout_secs:
        details.append(f"Response Timeout: {config.response_timeout_secs}s")

    return TextContent(type="text", text="\n".join(details))


@mcp.tool(
    description="""Create a new webhook tool for conversational AI agents.

    Webhook tools allow agents to make HTTP requests to external APIs during conversations.

    ⚠️ COST WARNING: Creating tools may affect your workspace configuration. Only use when explicitly requested.

    Args:
        name: The name of the tool (used by the LLM to identify when to call it)
        description: A detailed description of what the tool does and when it should be used
        url: The webhook URL. May include path parameters using {param} syntax, e.g., https://api.example.com/users/{user_id}
        method: HTTP method (GET, POST, PUT, DELETE, PATCH). Defaults to GET.
        request_headers: Optional dict of header name to header value for static headers
        path_params: Optional dict of path parameter names to their descriptions (for parameters in the URL like {user_id})
        query_params: Optional dict of query parameter names to their descriptions
        request_body_properties: Optional dict of body property names to their descriptions (for POST/PUT/PATCH requests)
        request_body_required: Optional list of required body property names
        response_timeout_secs: Optional timeout in seconds for the webhook response

    Returns:
        TextContent with the created tool details
    """
)
def create_webhook_tool(
    name: str,
    description: str,
    url: str,
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"] = "GET",
    request_headers: dict[str, str] | None = None,
    path_params: dict[str, str] | None = None,
    query_params: dict[str, str] | None = None,
    request_body_properties: dict[str, str] | None = None,
    request_body_required: list[str] | None = None,
    response_timeout_secs: int | None = None,
) -> TextContent:
    path_params_schema = None
    if path_params:
        path_params_schema = {
            param_name: LiteralJsonSchemaProperty(
                type="string",
                description=param_desc,
            )
            for param_name, param_desc in path_params.items()
        }

    query_params_schema = None
    if query_params:
        query_params_schema = QueryParamsJsonSchema(
            type="object",
            properties={
                param_name: LiteralJsonSchemaProperty(
                    type="string",
                    description=param_desc,
                )
                for param_name, param_desc in query_params.items()
            },
        )

    request_body_schema = None
    if request_body_properties:
        request_body_schema = ObjectJsonSchemaPropertyInput(
            type="object",
            properties={
                prop_name: LiteralJsonSchemaProperty(
                    type="string",
                    description=prop_desc,
                )
                for prop_name, prop_desc in request_body_properties.items()
            },
            required=request_body_required,
        )

    headers_dict = dict(request_headers) if request_headers else {}

    api_schema = WebhookToolApiSchemaConfigInput(
        url=url,
        method=method,
        request_headers=request_headers or {},        # Empty dict
        path_params_schema=path_params_schema or {},  # Empty dict
        query_params_schema=query_params_schema,      # Keep as None if not provided
        request_body_schema=request_body_schema,
    )

    tool_config = ToolRequestModelToolConfig_Webhook(
        name=name,
        description=description,
        api_schema=api_schema,
        response_timeout_secs=response_timeout_secs,
    )

    request = ToolRequestModel(tool_config=tool_config)

    try:
        response = client.conversational_ai.tools.create(request=request)
        return TextContent(
            type="text",
            text=f"Successfully created webhook tool.\nTool ID: {response.id}\nName: {name}\nURL: {url}\nMethod: {method}",
        )
    except Exception as e:
        make_error(f"Failed to create webhook tool: {str(e)}")
        return TextContent(type="text", text="")


@mcp.tool(
    description="""Create a new client tool for conversational AI agents.

    Client tools trigger events on the client side (e.g., in a web widget) rather than making server-side API calls.
    The client application must handle these tool calls.

    ⚠️ COST WARNING: Creating tools may affect your workspace configuration. Only use when explicitly requested.

    Args:
        name: The name of the tool (used by the LLM to identify when to call it)
        description: A detailed description of what the tool does and when it should be used
        expects_response: Whether the tool expects a response from the client. Defaults to False.
        parameters_properties: Optional dict of parameter names to their descriptions
        parameters_required: Optional list of required parameter names
        response_timeout_secs: Optional timeout in seconds for the client response

    Returns:
        TextContent with the created tool details
    """
)
def create_client_tool(
    name: str,
    description: str,
    expects_response: bool = False,
    parameters_properties: dict[str, str] | None = None,
    parameters_required: list[str] | None = None,
    response_timeout_secs: int | None = None,
) -> TextContent:
    parameters = None
    if parameters_properties:
        parameters = ObjectJsonSchemaPropertyInput(
            type="object",
            properties={
                prop_name: LiteralJsonSchemaProperty(
                    type="string",
                    description=prop_desc,
                )
                for prop_name, prop_desc in parameters_properties.items()
            },
            required=parameters_required,
        )

    tool_config = ToolRequestModelToolConfig_Client(
        name=name,
        description=description,
        expects_response=expects_response,
        parameters=parameters,
        response_timeout_secs=response_timeout_secs,
    )

    request = ToolRequestModel(tool_config=tool_config)

    try:
        response = client.conversational_ai.tools.create(request=request)
        return TextContent(
            type="text",
            text=f"Successfully created client tool.\nTool ID: {response.id}\nName: {name}\nExpects Response: {expects_response}",
        )
    except Exception as e:
        make_error(f"Failed to create client tool: {str(e)}")
        return TextContent(type="text", text="")


@mcp.tool(
    description="""Update an existing tool's configuration.

    ⚠️ COST WARNING: Updating tools may affect agents using this tool. Only use when explicitly requested.

    Args:
        tool_id: The ID of the tool to update
        name: Optional new name for the tool
        description: Optional new description for the tool
        url: Optional new URL (for webhook tools only)
        method: Optional new HTTP method (for webhook tools only)
        response_timeout_secs: Optional new timeout in seconds

    Returns:
        TextContent with the updated tool details
    """
)
def update_tool(
    tool_id: str,
    name: str | None = None,
    description: str | None = None,
    url: str | None = None,
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"] | None = None,
    response_timeout_secs: int | None = None,
) -> TextContent:
    """Update an existing tool's configuration."""
    existing_tool = client.conversational_ai.tools.get(tool_id=tool_id)
    existing_config = existing_tool.tool_config
    tool_type = getattr(existing_config, "type", "unknown")

    if tool_type == "webhook":
        existing_schema = getattr(existing_config, "api_schema", None)
        new_api_schema = WebhookToolApiSchemaConfigInput(
            url=url or getattr(existing_schema, "url", ""),
            method=method or getattr(existing_schema, "method", "GET"),
            request_headers=Schemas.transform(getattr(existing_schema, "request_headers", None)) or {},
            path_params_schema=Schemas.transform(getattr(existing_schema, "path_params_schema", None)) or {},
            query_params_schema=Schemas.convert(
                getattr(existing_schema, "query_params_schema", None),
                QueryParamsJsonSchema,
            ),
            request_body_schema=Schemas.convert(
                getattr(existing_schema, "request_body_schema", None),
                ObjectJsonSchemaPropertyInput,
            ),
        )

        tool_config = ToolRequestModelToolConfig_Webhook(
            name=name or existing_config.name,
            description=description or existing_config.description,
            api_schema=new_api_schema,
            response_timeout_secs=response_timeout_secs or getattr(existing_config, "response_timeout_secs", None),
        )

    elif tool_type == "client":
        tool_config = ToolRequestModelToolConfig_Client(
            name=name or existing_config.name,
            description=description or existing_config.description,
            expects_response=getattr(existing_config, "expects_response", False),
            parameters=Schemas.transform(getattr(existing_config, "parameters", None)),
            response_timeout_secs=response_timeout_secs or getattr(existing_config, "response_timeout_secs", None),
        )

    else:
        make_error(f"Cannot update tool of type: {tool_type}")
        return TextContent(type="text", text="")

    request = ToolRequestModel(tool_config=tool_config)

    try:
        response = client.conversational_ai.tools.update(tool_id=tool_id, request=request)
        return TextContent(
            type="text",
            text=f"Successfully updated tool.\nTool ID: {response.id}\nName: {response.tool_config.name}",
        )
    except Exception as e:
        make_error(f"Failed to update tool: {str(e)}")
        return TextContent(type="text", text="")


@mcp.tool(
    description="""Delete a tool from the workspace.

    ⚠️ WARNING: This action cannot be undone. Agents using this tool will lose access to it.

    Args:
        tool_id: The ID of the tool to delete

    Returns:
        TextContent confirming the deletion
    """
)
def delete_tool(tool_id: str) -> TextContent:
    try:
        dependent_agents = client.conversational_ai.tools.get_dependent_agents(tool_id=tool_id)
        if dependent_agents.agents and len(dependent_agents.agents) > 0:
            agent_names = ", ".join(agent.name for agent in dependent_agents.agents)
            return TextContent(
                type="text",
                text=f"Cannot delete tool. It is being used by the following agents: {agent_names}. Please remove the tool from these agents first.",
            )
    except Exception:
        pass

    try:
        client.conversational_ai.tools.delete(tool_id=tool_id)
        return TextContent(
            type="text",
            text=f"Successfully deleted tool with ID: {tool_id}",
        )
    except Exception as e:
        make_error(f"Failed to delete tool: {str(e)}")
        return TextContent(type="text", text="")


@mcp.tool(
    description="""Add a tool to a conversational AI agent.

    This attaches an existing workspace tool to an agent so the agent can use it during conversations.

    ⚠️ COST WARNING: This modifies agent configuration. Only use when explicitly requested.

    Args:
        agent_id: The ID of the agent to add the tool to
        tool_id: The ID of the tool to add (from list_workspace_tools or create_webhook_tool/create_client_tool)

    Returns:
        TextContent confirming the tool was added
    """
)
def add_tool_to_agent(agent_id: str, tool_id: str) -> TextContent:
    try:
        agent = client.conversational_ai.agents.get(agent_id=agent_id)

        existing_tool_ids = []
        if agent.conversation_config and agent.conversation_config.agent:
            prompt_config = agent.conversation_config.agent
            if hasattr(prompt_config, "tool_ids") and prompt_config.tool_ids:
                existing_tool_ids = list(prompt_config.tool_ids)

        if tool_id in existing_tool_ids:
            return TextContent(
                type="text",
                text=f"Tool {tool_id} is already attached to agent {agent_id}.",
            )

        existing_tool_ids.append(tool_id)

        client.conversational_ai.agents.update(
            agent_id=agent_id,
            conversation_config={
                "agent": {
                    "prompt": {
                        "tool_ids": existing_tool_ids,
                    }
                }
            },
        )

        return TextContent(
            type="text",
            text=f"Successfully added tool {tool_id} to agent {agent_id}.",
        )
    except Exception as e:
        make_error(f"Failed to add tool to agent: {str(e)}")
        return TextContent(type="text", text="")


@mcp.tool(
    description="""Remove a tool from a conversational AI agent.

    ⚠️ COST WARNING: This modifies agent configuration. Only use when explicitly requested.

    Args:
        agent_id: The ID of the agent to remove the tool from
        tool_id: The ID of the tool to remove

    Returns:
        TextContent confirming the tool was removed
    """
)
def remove_tool_from_agent(agent_id: str, tool_id: str) -> TextContent:
    try:
        agent = client.conversational_ai.agents.get(agent_id=agent_id)

        existing_tool_ids = []
        if agent.conversation_config and agent.conversation_config.agent:
            prompt = getattr(agent.conversation_config.agent, "prompt", None)
            if prompt and getattr(prompt, "tool_ids", None):
                existing_tool_ids = list(prompt.tool_ids)

        if tool_id not in existing_tool_ids:
            return TextContent(
                type="text",
                text=f"Tool {tool_id} is not attached to agent {agent_id}.",
            )

        existing_tool_ids.remove(tool_id)

        client.conversational_ai.agents.update(
            agent_id=agent_id,
            conversation_config={
                "agent": {
                    "prompt": {
                        "tool_ids": existing_tool_ids,
                    }
                }
            },
        )

        return TextContent(
            type="text",
            text=f"Successfully removed tool {tool_id} from agent {agent_id}.",
        )
    except Exception as e:
        make_error(f"Failed to remove tool from agent: {str(e)}")
        return TextContent(type="text", text="")


@mcp.tool(description="Get a list of agents that are using a specific tool")
def get_tool_dependent_agents(
    tool_id: str,
    page_size: int = 30,
) -> TextContent:
    """Get a list of agents that depend on a specific tool.

    Args:
        tool_id: The ID of the tool to check
        page_size: Number of agents to return (max 100, default 30)

    Returns:
        TextContent with a list of dependent agents
    """
    try:
        response = client.conversational_ai.tools.get_dependent_agents(
            tool_id=tool_id,
            page_size=min(page_size, 100),
        )

        if not response.agents:
            return TextContent(type="text", text="No agents are using this tool.")

        agents = ""
        for agent in response.agents:
            if getattr(agent, "type", "unknown") == "available":
                agents = "\n".join(f"- {agent.name} (ID: {agent.id})" for agent in response.agents)

        return TextContent(
            type="text",
            text=f"Agents using this tool ({len(response.agents)} total):\n{agents}",
        )
    except Exception as e:
        make_error(f"Failed to get dependent agents: {str(e)}")
        return TextContent(type="text", text="")
