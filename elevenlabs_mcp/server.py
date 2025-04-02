from datetime import datetime
import os
import base64
from typing import Literal
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.types import (
    TextContent,
    EmbeddedResource,
)
from elevenlabs.client import ElevenLabs
from elevenlabs_mcp.model import McpVoice
from elevenlabs_mcp.utils import (
    make_error,
    make_output_path,
    make_output_file,
    handle_input_file,
)
from elevenlabs_mcp.convai import create_conversation_config, create_platform_settings
from elevenlabs.types.knowledge_base_locator import KnowledgeBaseLocator
from io import BytesIO
from elevenlabs import play

load_dotenv()
api_key = os.getenv("ELEVENLABS_API_KEY")
base_path = os.getenv("ELEVENLABS_MCP_BASE_PATH")
DEFAULT_VOICE_ID = "9BWtsMINqrJLrRacOk9x"

if not api_key:
    raise ValueError("ELEVENLABS_API_KEY environment variable is required")

client = ElevenLabs(api_key=api_key)
mcp = FastMCP("ElevenLabs")


@mcp.tool(
    description="""Convert text to speech with a given voice and save the output audio file to a given directory.
    Directory is optional, if not provided, the output file will be saved to $HOME/Desktop.
    Only one of voice_id or voice_name can be provided. If no are provided, the default voice will be used."""
)
def text_to_speech(
    text: str,
    voice_name: str | None = None,
    output_directory: str | None = None,
    voice_id: str | None = None,
    stability: float = 0.5,
    similarity_boost: float = 0.75,
    style: float = 0,
    use_speaker_boost: bool = True,
    speed: float = 1.0,
):
    """Convert text to speech with a given voice and save the output audio file to a given directory.

    Args:
        text (str): The text to convert to speech.
        voice_name (str, optional): The name of the voice to use.
        stability (float, optional): Stability of the generated audio. Determines how stable the voice is and the randomness between each generation. Lower values introduce broader emotional range for the voice. Higher values can result in a monotonous voice with limited emotion.
        similarity_boost (float, optional): Similarity boost of the generated audio. Determines how closely the AI should adhere to the original voice when attempting to replicate it.
        style (float, optional): Style of the generated audio. Determines the style exaggeration of the voice. This setting attempts to amplify the style of the original speaker. It does consume additional computational resources and might increase latency if set to anything other than 0.
        use_speaker_boost (bool, optional): Use speaker boost of the generated audio. This setting boosts the similarity to the original speaker. Using this setting requires a slightly higher computational load, which in turn increases latency.
        speed (float, optional): Speed of the generated audio. Controls the speed of the generated speech. Values range from 0.7 to 1.2, with 1.0 being the default speed. Lower values create slower, more deliberate speech while higher values produce faster-paced speech. Extreme values can impact the quality of the generated speech.
        output_directory (str, optional): Directory where files should be saved.
            Defaults to $HOME/Desktop if not provided.

    Returns:
        List containing text content and audio data as embedded resource
    """
    if text == "":
        make_error("Text is required.")

    if voice_id is not None and voice_name is not None:
        make_error("voice_id and voice_name cannot both be provided.")

    voice = None
    if voice_id is not None:
        voice = client.voices.get(voice_id=voice_id)
    elif voice_name is not None:
        voices = client.voices.search(search=voice_name)
        if len(voices.voices) == 0:
            make_error("No voices found with that name.")
        voice = next((v for v in voices.voices if v.name == voice_name), None)
        if voice is None:
            make_error(f"Voice with name: {voice_name} does not exist.")

    voice_id = voice.voice_id if voice else DEFAULT_VOICE_ID

    output_path = make_output_path(output_directory, base_path)
    output_file_name = make_output_file("tts", text, output_path, "mp3")

    audio_data = client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
        voice_settings={
            "stability": stability,
            "similarity_boost": similarity_boost,
            "style": style,
            "use_speaker_boost": use_speaker_boost,
            "speed": speed,
        },
    )
    audio_bytes = b"".join(audio_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path / output_file_name, "wb") as f:
        f.write(audio_bytes)

    return TextContent(
        type="text",
        text=f"Success. File saved as: {output_path / output_file_name}. Voice used: {voice.name}",
    )


@mcp.tool(
    description="Transcribe speech from an audio file and either save the output text file to a given directory or return the text to the client directly."
)
def speech_to_text(
    input_file_path: str,
    language_code: str = "eng",
    diarize: bool = False,
    save_transcript_to_file: bool = True,
    return_transcript_to_client_directly: bool = False,
    output_directory: str | None = None,
) -> TextContent:
    """Transcribe speech from an audio file using ElevenLabs API.

    Args:
        file_path: Path to the audio file to transcribe
        language_code: Language code for transcription (default: "eng" for English)

    Returns:
        TextContent containing the transcription
    """
    if not save_transcript_to_file and not return_transcript_to_client_directly:
        make_error("Must save transcript to file or return it to the client directly.")
    file_path = handle_input_file(input_file_path)
    if save_transcript_to_file:
        output_path = make_output_path(output_directory, base_path)
        output_file_name = make_output_file("stt", file_path.name, output_path, "txt")
    with file_path.open("rb") as f:
        audio_bytes = f.read()
    transcription = client.speech_to_text.convert(
        model_id="scribe_v1",
        file=audio_bytes,
        language_code=language_code,
        enable_logging=True,
        diarize=diarize,
        tag_audio_events=True,
    )

    if save_transcript_to_file:
        with open(output_path / output_file_name, "w") as f:
            f.write(transcription.text)

    if return_transcript_to_client_directly:
        return TextContent(type="text", text=transcription.text)
    else:
        return TextContent(
            type="text", text=f"Transcription saved to {output_path / output_file_name}"
        )


@mcp.tool(
    description="""Convert text description of a sound effect to sound effect with a given duration and save the output audio file to a given directory. 
    Directory is optional, if not provided, the output file will be saved to $HOME/Desktop.
    Duration must be between 0.5 and 5 seconds."""
)
def text_to_sound_effects(
    text: str, duration_seconds: float = 2.0, output_directory: str | None = None
) -> list[TextContent | EmbeddedResource]:
    if duration_seconds < 0.5 or duration_seconds > 5:
        make_error("Duration must be between 0.5 and 5 seconds")
    output_path = make_output_path(output_directory, base_path)
    output_file_name = make_output_file("sfx", text, output_path, "mp3")

    audio_data = client.text_to_sound_effects.convert(
        text=text,
        output_format="mp3_44100_128",
        duration_seconds=duration_seconds,
    )
    audio_bytes = b"".join(audio_data)

    with open(output_path / output_file_name, "wb") as f:
        f.write(audio_bytes)

    return TextContent(
        type="text",
        text=f"Success. File saved as: {output_path / output_file_name}",
    )


@mcp.tool(description="List all available voices")
def list_voices() -> list[McpVoice]:
    """List all available voices.

    Returns:
        A formatted list of available voices with their IDs and names
    """
    response = client.voices.get_all()
    return [
        McpVoice(id=voice.voice_id, name=voice.name, category=voice.category)
        for voice in response.voices
    ]


@mcp.resource("voices://list")
def get_voices() -> list[McpVoice]:
    """Get a list of all available voices."""
    response = client.voices.get_all()
    return [
        McpVoice(id=voice.voice_id, name=voice.name, category=voice.category)
        for voice in response.voices
    ]


@mcp.tool(description="Get details of a specific voice.")
@mcp.tool(
    description="Search for voices by search term. Returns all voices if no search term is provided. Searches in name, description, labels and category."
)
def search_voices(
    search: str | None = None,
    sort: Literal["created_at_unix", "name"] = "name",
    sort_direction: Literal["asc", "desc"] = "desc",
) -> list[McpVoice]:
    """Search for voices.

    Args:
        search: Search term to filter voices by. Searches in name, description, labels and category.
        sort: Which field to sort by. `created_at_unix` might not be available for older voices.
        sort_direction: Sort order, either ascending or descending.

    Returns:
        List of voices that match the search criteria.
    """
    response = client.voices.search(
        search=search, sort=sort, sort_direction=sort_direction
    )
    return [
        McpVoice(id=voice.voice_id, name=voice.name, category=voice.category)
        for voice in response.voices
    ]


@mcp.resource("voice://{voice_id}")
def get_voice(voice_id: str) -> McpVoice:
    """Get details of a specific voice."""
    response = client.voices.get(voice_id=voice_id)
    return McpVoice(
        id=response.voice_id,
        name=response.name,
        category=response.category,
        fine_tuning_status=response.fine_tuning.state,
    )


@mcp.tool(description="Clone a voice using provided audio files")
def voice_clone(
    name: str, files: list[str], description: str | None = None
) -> TextContent:
    input_files = [str(handle_input_file(file).absolute()) for file in files]
    voice = client.clone(name=name, description=description, files=input_files)
    return TextContent(
        type="text",
        text=f"""Voice cloned successfully: Name: {voice.name}
        ID: {voice.voice_id}
        Category: {voice.category}
        Description: {voice.description or "N/A"}""",
    )


@mcp.tool(
    description="""Isolate audio from a file and save the output audio file to a given directory. 
    Directory is optional, if not provided, the output file will be saved to $HOME/Desktop."""
)
def isolate_audio(
    input_file_path: str, output_directory: str | None = None
) -> list[TextContent | EmbeddedResource]:
    file_path = handle_input_file(input_file_path)
    output_path = make_output_path(output_directory, base_path)
    output_file_name = make_output_file("iso", file_path.name, output_path, "mp3")
    with file_path.open("rb") as f:
        audio_bytes = f.read()
    audio_data = client.audio_isolation.audio_isolation(
        audio=audio_bytes,
    )
    audio_bytes = b"".join(audio_data)

    with open(output_path / output_file_name, "wb") as f:
        f.write(audio_bytes)

    return TextContent(
        type="text",
        text=f"Success. File saved as: {output_path / output_file_name}",
    )


@mcp.tool(
    description="Check the current subscription status. Could be used to measure the usage of the API."
)
def check_subscription() -> TextContent:
    subscription = client.user.get_subscription()
    return TextContent(type="text", text=f"{subscription.model_dump_json(indent=2)}")


@mcp.tool(description="Create a conversational AI agent with custom configuration")
def create_agent(
    name: str,
    system_prompt: str,
    voice_id: str | None = None,
    language: str = "en",
    llm: str = "gemini-2.0-flash-001",
    first_message: str | None = None,
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

    response = client.conversational_ai.create_agent(
        name=name,
        conversation_config=conversation_config,
        platform_settings=platform_settings,
    )

    return TextContent(
        type="text",
        text=f"""Agent created successfully: Name: {name}, Agent ID: {response.agent_id}, System Prompt: {system_prompt}, Voice ID: {voice_id or "Default"}, Language: {language}, LLM: {llm}, You can use this agent ID for future interactions with the agent.""",
    )


@mcp.tool(
    description="Add a knowledge base to ElevenLabs workspace. Allowed types are epub, pdf, docx, txt, html."
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

    if text is not None:
        text_bytes = text.encode("utf-8")
        text_io = BytesIO(text_bytes)
        text_io.name = "text.txt"
        text_io.content_type = "text/plain"
        file = text_io
    elif input_file_path is not None:
        path = handle_input_file(file_path=input_file_path, audio_content_check=False)
        file = open(path, "rb")

    response = client.conversational_ai.add_to_knowledge_base(
        name=knowledge_base_name,
        url=url,
        file=file,
    )
    agent = client.conversational_ai.get_agent(agent_id=agent_id)
    agent.conversation_config.agent.prompt.knowledge_base.append(
        KnowledgeBaseLocator(
            type="file" if file else "url",
            name=knowledge_base_name,
            id=response.id,
        )
    )
    client.conversational_ai.update_agent(
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
    response = client.conversational_ai.get_agents()

    if not response.agents:
        return TextContent(type="text", text="No agents found.")

    agent_list = ",".join(
        f"{agent.name} (ID: {agent.agent_id})" for agent in response.agents
    )

    return TextContent(type="text", text=f"Available agents: {agent_list}")


@mcp.tool(description="Get details about a specific conversational AI agent")
def get_agent(agent_id: str) -> TextContent:
    """Get details about a specific conversational AI agent.

    Args:
        agent_id: The ID of the agent to retrieve

    Returns:
        TextContent with detailed information about the agent
    """
    response = client.conversational_ai.get_agent(agent_id=agent_id)

    voice_info = "None"
    if response.conversation_config.tts:
        voice_info = f"Voice ID: {response.conversation_config.tts.voice_id}"

    return TextContent(
        type="text",
        text=f"Agent Details: Name: {response.name}, Agent ID: {response.agent_id}, Voice Configuration: {voice_info}, Created At: {datetime.fromtimestamp(response.metadata.created_at_unix_secs).strftime("%Y-%m-%d %H:%M:%S")}",
    )


@mcp.tool(
    description="Transform audio from one voice to another using provided audio files"
)
def speech_to_speech(
    input_file_path: str,
    voice_name: str = "Adam",
    output_directory: str = "",
) -> TextContent:
    voices = client.voices.search(search=voice_name)

    if len(voices.voices) == 0:
        make_error("No voice found with that name.")

    voice = next((v for v in voices.voices if v.name == voice_name), None)

    if voice is None:
        make_error(f"Voice with name: {voice_name} does not exist.")

    file_path = handle_input_file(input_file_path)
    output_path = make_output_path(output_directory, base_path)
    output_file_name = make_output_file("sts", file_path.name, output_path, "mp3")

    with file_path.open("rb") as f:
        audio_bytes = f.read()

    audio_data = client.speech_to_speech.convert(
        model_id="eleven_english_sts_v2",
        voice_id=voice.voice_id,
        audio=audio_bytes,
    )

    audio_bytes = b"".join(audio_data)

    with open(output_path / output_file_name, "wb") as f:
        f.write(audio_bytes)

    return TextContent(
        type="text", text=f"Success. File saved as: {output_path / output_file_name}"
    )


@mcp.tool(
    description="Create voice previews from a text prompt. Creates three previews with slight variations. Saves the previews to a given directory. If no text is provided, the tool will auto-generate text."
)
def text_to_voice(
    voice_description: str,
    text: str | None = None,
    output_directory: str = "",
) -> TextContent:
    if voice_description == "":
        make_error("Voice description is required.")

    previews = client.text_to_voice.create_previews(
        voice_description=voice_description,
        text=text,
        auto_generate_text=True if text is None else False,
    )

    output_path = make_output_path(output_directory, base_path)

    generated_voice_ids = []

    for preview in previews.previews:
        output_file_name = make_output_file(
            "voice_design", preview.generated_voice_id, output_path, "mp3", full_id=True
        )
        generated_voice_ids.append(preview.generated_voice_id)
        audio_bytes = base64.b64decode(preview.audio_base_64)

        with open(output_path / output_file_name, "wb") as f:
            f.write(audio_bytes)

    return TextContent(
        type="text",
        text=f"Success. Files saved at: {output_path}. Generated voice IDs are: {', '.join(generated_voice_ids)}",
    )


@mcp.tool(
    description="Add a generated voice to the voice library. Uses the voice ID from the `text_to_voice` tool."
)
def create_voice_from_preview(
    generated_voice_id: str,
    voice_name: str,
    voice_description: str,
) -> TextContent:
    voice = client.text_to_voice.create_voice_from_preview(
        voice_name=voice_name,
        voice_description=voice_description,
        generated_voice_id=generated_voice_id,
    )

    return TextContent(
        type="text",
        text=f"Success. Voice created: {voice.name} with ID:{voice.voice_id}",
    )


@mcp.tool(description="Play an audio file. Supports WAV and MP3 formats.")
def play_audio(input_file_path: str) -> TextContent:
    file_path = handle_input_file(input_file_path)
    play(open(file_path, "rb").read(), use_ffmpeg=False)
    return TextContent(type="text", text=f"Successfully played audio file: {file_path}")


if __name__ == "__main__":
    mcp.run()
