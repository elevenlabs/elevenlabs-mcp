from pathlib import Path
from unittest.mock import MagicMock, patch

from elevenlabs_mcp import server


def test_voice_clone_uploads_file_bytes_not_paths(tmp_path):
    """Regression test for #62: voice_clone must upload file contents, not path strings."""
    audio = tmp_path / "sample.mp3"
    audio.write_bytes(b"fake-audio-bytes")

    mock_client = MagicMock()
    voice = mock_client.voices.ivc.create.return_value
    voice.name, voice.voice_id, voice.category, voice.description = (
        "My Clone",
        "abc123",
        "cloned",
        None,
    )

    with (
        patch.object(server, "client", mock_client),
        patch.object(server, "handle_input_file", lambda file: Path(file)),
    ):
        server.voice_clone(name="My Clone", files=[str(audio)])

    _, kwargs = mock_client.voices.ivc.create.call_args
    assert kwargs["files"] == [b"fake-audio-bytes"]
    assert all(isinstance(f, bytes) for f in kwargs["files"])
