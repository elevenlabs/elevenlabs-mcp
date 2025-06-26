def create_conversation_config(
    language: str,
    system_prompt: str,
    llm: str,
    first_message: str | None,
    temperature: float,
    max_tokens: int | None,
    asr_quality: str,
    voice_id: str | None,
    model_id: str,
    optimize_streaming_latency: int,
    stability: float,
    similarity_boost: float,
    turn_timeout: int,
    max_duration_seconds: int,
    tools: list | None = None,
) -> dict:
    return {
        "agent": {
            "language": language,
            "prompt": {
                "prompt": system_prompt,
                "llm": llm,
                "tools": tools if tools is not None else [{"type": "system", "name": "end_call", "description": ""}],
                "knowledge_base": [],
                "temperature": temperature,
                **({"max_tokens": max_tokens} if max_tokens else {}),
            },
            **({"first_message": first_message} if first_message else {}),
            "dynamic_variables": {"dynamic_variable_placeholders": {}},
        },
        "asr": {
            "quality": asr_quality,
            "provider": "elevenlabs",
            "user_input_audio_format": "pcm_16000",
            "keywords": [],
        },
        "tts": {
            **({"voice_id": voice_id} if voice_id else {}),
            "model_id": model_id,
            "agent_output_audio_format": "pcm_16000",
            "optimize_streaming_latency": optimize_streaming_latency,
            "stability": stability,
            "similarity_boost": similarity_boost,
        },
        "turn": {"turn_timeout": turn_timeout},
        "conversation": {
            "max_duration_seconds": max_duration_seconds,
            "client_events": [
                "audio",
                "interruption",
                "user_transcript",
                "agent_response",
                "agent_response_correction",
            ],
        },
        "language_presets": {},
        "is_blocked_ivc": False,
        "is_blocked_non_ivc": False,
    }


def create_platform_settings(
    record_voice: bool,
    retention_days: int,
) -> dict:
    return {
        "widget": {
            "variant": "full",
            "avatar": {"type": "orb", "color_1": "#6DB035", "color_2": "#F5CABB"},
            "feedback_mode": "during",
            "terms_text": '#### Terms and conditions\n\nBy clicking "Agree," and each time I interact with this AI agent, I consent to the recording, storage, and sharing of my communications with third-party service providers, and as described in the Privacy Policy.\nIf you do not wish to have your conversations recorded, please refrain from using this service.',
            "show_avatar_when_collapsed": True,
        },
        "evaluation": {},
        "auth": {"allowlist": []},
        "overrides": {},
        "call_limits": {"agent_concurrency_limit": -1, "daily_limit": 100000},
        "privacy": {
            "record_voice": record_voice,
            "retention_days": retention_days,
            "delete_transcript_and_pii": True,
            "delete_audio": True,
            "apply_to_existing_conversations": False,
        },
        "data_collection": {},
    }
