"""
ElevenLabs MCP Server - Conversational AI Agents and Tools Management

This module contains tools for managing conversational AI agents and their
server tools (webhooks and client tools).
"""

from datetime import datetime
from typing_extensions import TypedDict, NotRequired
from io import BytesIO

from elevenlabs.types import (
    AgentConfig,
    ConversationalConfig,
    DynamicVariableAssignment,
    DynamicVariablesConfig,
    DynamicVariablesConfigDynamicVariablePlaceholdersValue,
    LiteralJsonSchemaProperty,
    ObjectJsonSchemaPropertyInput,
    PromptAgentApiModelOutput,
    QueryParamsJsonSchema,
    ToolRequestModel,
    ToolRequestModelToolConfig_Client,
    ToolRequestModelToolConfig_Webhook,
    WebhookToolApiSchemaConfigInput,
    WebhookToolApiSchemaConfigInputMethod,
    WebhookToolApiSchemaConfigInputRequestHeadersValue,
)
from elevenlabs.types.knowledge_base_locator import KnowledgeBaseLocator
from elevenlabs_mcp.convai import create_conversation_config, create_platform_settings
from elevenlabs_mcp.mcp import mcp, client, DEFAULT_VOICE_ID
from elevenlabs_mcp.utils import make_error, handle_input_file
from mcp.types import TextContent


class WebhookConfig(TypedDict):
    """Configuration for webhook tool creation.

    Attributes:
        name: The name of the tool (used by the LLM to identify when to call it)
        description: A detailed description of what the tool does
        url: The webhook URL. May include path parameters using {param} syntax
        method: HTTP method (GET, POST, PUT, DELETE, PATCH). Defaults to GET.
        response_timeout_secs: Optional timeout in seconds for the webhook response
        disable_interruptions: Whether to disable interruptions while tool is running
        force_pre_tool_speech: Force agent speech before tool execution
    """
    url: str
    name: str
    description: str
    method: NotRequired[WebhookToolApiSchemaConfigInputMethod]
    response_timeout_secs: NotRequired[int]
    disable_interruptions: NotRequired[bool]
    force_pre_tool_speech: NotRequired[bool]


class ClientToolConfig(TypedDict):
    """Configuration for client tool creation.

    Attributes:
        name: The name of the tool (used by the LLM to identify when to call it)
        description: A detailed description of what the tool does
        expects_response: Whether the tool expects a response from the client
        response_timeout_secs: Optional timeout in seconds for the client response
        disable_interruptions: Whether to disable interruptions while tool is running
        force_pre_tool_speech: Force agent speech before tool execution
    """
    name: str
    description: str
    expects_response: NotRequired[bool]
    response_timeout_secs: NotRequired[int]
    disable_interruptions: NotRequired[bool]
    force_pre_tool_speech: NotRequired[bool]


class UpdateWebhookConfig(TypedDict):
    """Configuration for updating a webhook tool. All fields are optional.

    Attributes:
        name: New name for the tool
        description: New description for the tool
        url: New webhook URL
        method: New HTTP method
        response_timeout_secs: New timeout in seconds
        disable_interruptions: Whether to disable interruptions
        force_pre_tool_speech: Force agent speech before execution
    """
    url: NotRequired[str]
    name: NotRequired[str]
    description: NotRequired[str]
    method: NotRequired[WebhookToolApiSchemaConfigInputMethod]
    response_timeout_secs: NotRequired[int]
    disable_interruptions: NotRequired[bool]
    force_pre_tool_speech: NotRequired[bool]


class UpdateClientToolConfig(TypedDict):
    """Configuration for updating a client tool. All fields are optional.

    Attributes:
        name: New name for the tool
        description: New description for the tool
        expects_response: Whether the tool expects a response
        response_timeout_secs: New timeout in seconds
        disable_interruptions: Whether to disable interruptions
        force_pre_tool_speech: Force agent speech before execution
    """
    name: NotRequired[str]
    description: NotRequired[str]
    expects_response: NotRequired[bool]
    response_timeout_secs: NotRequired[int]
    disable_interruptions: NotRequired[bool]
    force_pre_tool_speech: NotRequired[bool]


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
    voice_info = "None"
    response = client.conversational_ai.agents.get(agent_id=agent_id)
    if response.conversation_config.tts:
        voice_info = f"Voice ID: {response.conversation_config.tts.voice_id}"

    ts = datetime.fromtimestamp(response.metadata.created_at_unix_secs)\
                 .strftime('%Y-%m-%d %H:%M:%S')
    return TextContent(
        type="text",
        text="Agent Details: " +
            f"Name: {response.name}, " +
            f"Agent ID: {response.agent_id}, " +
            f"Voice Configuration: {voice_info}, " +
            f"Created At: {ts}",
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
    tool = client.conversational_ai.tools.get(tool_id=tool_id)
    config = tool.tool_config
    schema = getattr(config, "api_schema")

    details = [
        f"Tool ID: {tool.id}",
        f"Type: {getattr(config, 'type', 'unknown')}",
        f"Name: {getattr(config, 'name', 'unnamed')}",
    ]

    if desc := getattr(config, "description", None):
        details.append(f"Description: {desc}")

    if getattr(config, "type", "unknown") == "webhook" and schema:
        details.append(f"URL: {getattr(schema, 'url', '')}")
        if method := getattr(schema, "method", None):
            details.append(f"Method: {method}")

    if timeout := getattr(config, "response_timeout_secs", None):
        details.append(f"Response Timeout: {timeout}s")

    return TextContent(type="text", text="\n".join(details))


@mcp.tool(
    description="""Create a new webhook tool for conversational AI agents.

    Webhook tools allow agents to make HTTP requests to external APIs during conversations.

    ⚠️ COST WARNING: Creating tools may affect your workspace configuration. Only use when explicitly requested.

    Args:
        config: WebhookConfig with basic settings (name, description, url, method, timeouts, execution options)
        headers: Optional dict of header name to WebhookToolApiSchemaConfigInputRequestHeadersValue (str, secret, or dynamic var)
        urlparams: Optional dict of URL path parameter names to LiteralJsonSchemaProperty (for {param} in URL)
        qparams: Optional QueryParamsJsonSchema for query parameters
        body: Optional ObjectJsonSchemaPropertyInput for request body schema
        dynamic_vars: Optional dict of dynamic variable names to placeholder values
        dynamic_vars_assign: Optional list of DynamicVariableAssignment for extracting values from response

    Returns:
        TextContent with the created tool details
    """
)
def create_webhook_tool(
    config: WebhookConfig,
    headers: dict[str, WebhookToolApiSchemaConfigInputRequestHeadersValue] | None = None,
    urlparams: dict[str, LiteralJsonSchemaProperty] | None = None,
    qparams: QueryParamsJsonSchema | None = None,
    body: ObjectJsonSchemaPropertyInput | None = None,
    dynamic_vars: dict[str, DynamicVariablesConfigDynamicVariablePlaceholdersValue] | None = None,
    dynamic_vars_assign: list[DynamicVariableAssignment] | None = None,
) -> TextContent:
    try:
        tool = client.conversational_ai.tools.create(
            request=ToolRequestModel(
                tool_config=ToolRequestModelToolConfig_Webhook(
                    name=config["name"],
                    description=config["description"],
                    api_schema=WebhookToolApiSchemaConfigInput(
                        url=config["url"],
                        method=config.get("method", "GET"),
                        request_headers=headers,
                        path_params_schema=urlparams,
                        query_params_schema=qparams,
                        request_body_schema=body,
                    ),
                    response_timeout_secs=config.get("response_timeout_secs"),
                    disable_interruptions=config.get("disable_interruptions"),
                    force_pre_tool_speech=config.get("force_pre_tool_speech"),
                    dynamic_variables=DynamicVariablesConfig(dynamic_variable_placeholders=dynamic_vars) if dynamic_vars else None,
                    assignments=dynamic_vars_assign,
                )
            )
        )
        return TextContent(
            type="text",
            text="Successfully created webhook tool." +
                 f"\nTool ID: {tool.id}\nName: {config['name']}\nURL: {config['url']}\nMethod: {config.get('method', 'GET')}",
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
        config: ClientToolConfig with basic settings (name, description, expects_response, timeouts, execution options)
        parameters: Optional ObjectJsonSchemaPropertyInput for tool parameters schema
        dynamic_vars: Optional dict of dynamic variable names to placeholder values
        dynamic_vars_assign: Optional list of DynamicVariableAssignment for extracting values from response

    Returns:
        TextContent with the created tool details
    """
)
def create_client_tool(
    config: ClientToolConfig,
    parameters: ObjectJsonSchemaPropertyInput | None = None,
    dynamic_vars: dict[str, DynamicVariablesConfigDynamicVariablePlaceholdersValue] | None = None,
    dynamic_vars_assign: list[DynamicVariableAssignment] | None = None,
) -> TextContent:
    try:
        tool = client.conversational_ai.tools.create(
            request=ToolRequestModel(
                tool_config=ToolRequestModelToolConfig_Client(
                    name=config["name"],
                    description=config["description"],
                    expects_response=config.get("expects_response", False),
                    parameters=parameters,
                    response_timeout_secs=config.get("response_timeout_secs"),
                    disable_interruptions=config.get("disable_interruptions"),
                    force_pre_tool_speech=config.get("force_pre_tool_speech"),
                    dynamic_variables=DynamicVariablesConfig(dynamic_variable_placeholders=dynamic_vars) if dynamic_vars else None,
                    assignments=dynamic_vars_assign,
                )
            )
        )
        return TextContent(
            type="text",
            text="Successfully created client tool." +
                 f"\nTool ID: {tool.id}\nName: {config['name']}" +
                 f"\nExpects Response: {config.get('expects_response', False)}",
        )
    except Exception as e:
        make_error(f"Failed to create client tool: {str(e)}")
        return TextContent(type="text", text="")


@mcp.tool(
    description="""Update an existing webhook tool's configuration.

    ⚠️ COST WARNING: Updating tools may affect agents using this tool. Only use when explicitly requested.

    Args:
        tool_id: The ID of the webhook tool to update
        config: UpdateWebhookConfig with fields to update (all optional)
        headers: Optional new headers dict
        urlparams: Optional new path params schema
        qparams: Optional new query params schema
        body: Optional new body schema
        dynamic_vars: Optional new dynamic variables
        dynamic_vars_assign: Optional new dynamic variable assignments

    Returns:
        TextContent with the updated tool details
    """
)
def update_webhook_tool(
    tool_id: str,
    config: UpdateWebhookConfig | None = None,
    headers: dict[str, WebhookToolApiSchemaConfigInputRequestHeadersValue] | None = None,
    urlparams: dict[str, LiteralJsonSchemaProperty] | None = None,
    qparams: QueryParamsJsonSchema | None = None,
    body: ObjectJsonSchemaPropertyInput | None = None,
    dynamic_vars: dict[str, DynamicVariablesConfigDynamicVariablePlaceholdersValue] | None = None,
    dynamic_vars_assign: list[DynamicVariableAssignment] | None = None,
) -> TextContent:
    config = config or {}
    curr = client.conversational_ai.tools.get(tool_id=tool_id)
    schema = getattr(curr.tool_config, "api_schema", None)

    if getattr(curr.tool_config, "type") != "webhook":
        make_error(f"Tool {tool_id} is not a webhook tool")

    try:
        modified = client.conversational_ai.tools.update(
            tool_id=tool_id,
            request=ToolRequestModel(
                tool_config=ToolRequestModelToolConfig_Webhook(
                    name=config.get("name", getattr(curr.tool_config, "name")),
                    description=config.get("description", getattr(curr.tool_config, "description")),
                    api_schema=WebhookToolApiSchemaConfigInput(
                        url=config.get("url", getattr(schema, "url", "")),
                        method=config.get("method", getattr(schema, "method", "GET")),
                        request_headers=headers or getattr(schema, "request_headers"),
                        path_params_schema=urlparams or getattr(schema, "path_params_schema"),
                        query_params_schema=qparams or getattr(schema, "query_params_schema"),
                        request_body_schema=body or getattr(schema, "request_body_schema"),
                    ),
                    response_timeout_secs=(
                        config.get("response_timeout_secs")
                        if "response_timeout_secs" in config
                        else getattr(curr.tool_config, "response_timeout_secs")
                    ),
                    disable_interruptions=(
                        config.get("disable_interruptions")
                        if "disable_interruptions" in config
                        else getattr(curr.tool_config, "disable_interruptions")
                    ),
                    force_pre_tool_speech=(
                        config.get("force_pre_tool_speech")
                        if "force_pre_tool_speech" in config
                        else getattr(curr.tool_config, "force_pre_tool_speech")
                    ),
                    dynamic_variables=(
                        DynamicVariablesConfig(dynamic_variable_placeholders=dynamic_vars)
                        if dynamic_vars is not None
                        else getattr(curr.tool_config, "dynamic_variables")
                    ),
                    assignments=dynamic_vars_assign or getattr(curr.tool_config, "assignments"),
                )
            ),
        )
        return TextContent(
            type="text",
            text="Successfully updated webhook tool." +
                 f"\nTool ID: {modified.id}\nName: {getattr(modified.tool_config, 'name')}",
        )
    except Exception as e:
        make_error(f"Failed to update webhook tool: {str(e)}")
        return TextContent(type="text", text="")


@mcp.tool(
    description="""Update an existing client tool's configuration.

    ⚠️ COST WARNING: Updating tools may affect agents using this tool. Only use when explicitly requested.

    Args:
        tool_id: The ID of the client tool to update
        config: UpdateClientToolConfig with fields to update (all optional)
        parameters: Optional new parameters schema
        dynamic_vars: Optional new dynamic variables
        dynamic_vars_assign: Optional new dynamic variable assignments

    Returns:
        TextContent with the updated tool details
    """
)
def update_client_tool(
    tool_id: str,
    config: UpdateClientToolConfig | None = None,
    parameters: ObjectJsonSchemaPropertyInput | None = None,
    dynamic_vars: dict[str, DynamicVariablesConfigDynamicVariablePlaceholdersValue] | None = None,
    dynamic_vars_assign: list[DynamicVariableAssignment] | None = None,
) -> TextContent:
    config = config or {}
    curr = client.conversational_ai.tools.get(tool_id=tool_id)

    if getattr(curr.tool_config, "type") != "client":
        make_error(f"Tool {tool_id} is not a client tool")

    try:
        modified = client.conversational_ai.tools.update(
            tool_id=tool_id,
            request=ToolRequestModel(
                tool_config=ToolRequestModelToolConfig_Client(
                    name=config.get("name", getattr(curr.tool_config, "name")),
                    description=config.get("description", getattr(curr.tool_config, "description")),
                    parameters=parameters or getattr(curr.tool_config, "parameters"),
                    expects_response=(
                        config.get("expects_response")
                        if "expects_response" in config
                        else getattr(curr.tool_config, "expects_response", False)
                    ),
                    response_timeout_secs=(
                        config.get("response_timeout_secs")
                        if "response_timeout_secs" in config
                        else getattr(curr.tool_config, "response_timeout_secs")
                    ),
                    disable_interruptions=(
                        config.get("disable_interruptions")
                        if "disable_interruptions" in config
                        else getattr(curr.tool_config, "disable_interruptions")
                    ),
                    force_pre_tool_speech=(
                        config.get("force_pre_tool_speech")
                        if "force_pre_tool_speech" in config
                        else getattr(curr.tool_config, "force_pre_tool_speech")
                    ),
                    dynamic_variables=(
                        DynamicVariablesConfig(dynamic_variable_placeholders=dynamic_vars)
                        if dynamic_vars is not None
                        else getattr(curr.tool_config, "dynamic_variables")
                    ),
                    assignments=dynamic_vars_assign if dynamic_vars_assign else getattr(curr.tool_config, "assignments"),
                )
            ),
        )
        return TextContent(
            type="text",
            text="Successfully updated client tool." +
                 f"\nTool ID: {modified.id}\nName: {getattr(modified.tool_config, 'name')}",
        )
    except Exception as e:
        make_error(f"Failed to update client tool: {str(e)}")
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
        deps = client.conversational_ai.tools.get_dependent_agents(tool_id=tool_id)
        if deps.agents and len(deps.agents) > 0:
            return TextContent(
                type="text",
                text="Tool currently in use by the following agents: " +
                    f"{', '.join(agent.name for agent in deps.agents)}. " +
                    "Please remove the tool from these agents first.",
            )

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
    agent = client.conversational_ai.agents.get(agent_id=agent_id)
    prompt = getattr(agent.conversation_config.agent, "prompt")
    tool_ids = list(getattr(prompt, "tool_ids") or [])

    if tool_id in tool_ids:
        return TextContent(
            type="text",
            text=f"Tool {tool_id} is already attached to agent {agent_id}.",
        )

    tool_ids.append(tool_id)

    try:
        client.conversational_ai.agents.update(
            agent_id=agent_id,
            conversation_config=ConversationalConfig(
                agent=AgentConfig(
                    prompt=PromptAgentApiModelOutput(tool_ids=tool_ids)
                )
            ),
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
    agent = client.conversational_ai.agents.get(agent_id=agent_id)
    prompt = getattr(getattr(agent.conversation_config, "agent", None), "prompt", None)
    tool_ids: list[str] = list(getattr(prompt, "tool_ids", None) or [])

    if tool_id not in tool_ids:
        return TextContent(
            type="text",
            text=f"Tool {tool_id} is not attached to agent {agent_id}.",
        )

    tool_ids.remove(tool_id)

    try:
        client.conversational_ai.agents.update(
            agent_id=agent_id,
            conversation_config=ConversationalConfig(
                agent=AgentConfig(
                    prompt=PromptAgentApiModelOutput(tool_ids=tool_ids)
                )
            ),
        )
    except Exception as e:
        make_error(f"Failed to remove tool from agent: {str(e)}")

    return TextContent(
        type="text",
        text=f"Successfully removed tool {tool_id} from agent {agent_id}.",
    )


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
        deps = client.conversational_ai.tools.get_dependent_agents(
            tool_id=tool_id,
            page_size=min(page_size, 100)
        )
        if not deps.agents:
            return TextContent(type="text", text="No agents are using this tool.")

        agents = ""
        for agent in deps.agents:
            if getattr(agent, "type", "unknown") == "available":
                agents = "\n".join(
                    f"- {getattr(agent, 'name', 'Unknown')} (ID: {getattr(agent, 'id', 'Unknown')})"
                    for agent in deps.agents
                )

        return TextContent(
            type="text",
            text=f"Agents using this tool ({len(deps.agents)} total):\n{agents}",
        )
    except Exception as e:
        make_error(f"Failed to get dependent agents: {str(e)}")
        return TextContent(type="text", text="")
