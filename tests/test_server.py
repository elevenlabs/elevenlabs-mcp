import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("ELEVENLABS_API_KEY", "test-key")

from elevenlabs_mcp import server  # noqa: E402
from elevenlabs_mcp.utils import ElevenLabsMcpError  # noqa: E402


def test_voice_clone_passes_open_file_objects_with_real_bytes(tmp_path):
    audio_file_1 = tmp_path / "voice1.mp3"
    audio_file_2 = tmp_path / "voice2.mp3"
    audio_file_1.write_bytes(b"FAKEAUDIO1")
    audio_file_2.write_bytes(b"FAKEAUDIO2")
    read_payloads = []

    def fake_create(name, description, files):
        assert all(not isinstance(file, str) for file in files)
        assert all(hasattr(file, "read") for file in files)
        for file in files:
            read_payloads.append(file.read())
        return SimpleNamespace(
            name=name,
            voice_id="vid",
            category="cloned",
            description=description,
        )

    with patch.object(server, "client") as mock_client:
        assert isinstance(mock_client, MagicMock)
        mock_client.voices.ivc.create.side_effect = fake_create

        result = server.voice_clone(
            name="Test Voice",
            files=[str(audio_file_1.resolve()), str(audio_file_2.resolve())],
            description="sample description",
        )

    assert read_payloads == [b"FAKEAUDIO1", b"FAKEAUDIO2"]
    assert "Voice cloned successfully" in result.text
    assert "ID: vid" in result.text


def test_voice_clone_single_file_closes_handle_after_call(tmp_path):
    audio_file = tmp_path / "voice.mp3"
    audio_file.write_bytes(b"FAKEAUDIO")
    captured_handles = []

    def fake_create(name, description, files):
        captured_handles.extend(files)
        return SimpleNamespace(
            name=name,
            voice_id="vid",
            category="cloned",
            description=description,
        )

    with patch.object(server, "client") as mock_client:
        assert isinstance(mock_client, MagicMock)
        mock_client.voices.ivc.create.side_effect = fake_create

        server.voice_clone(
            name="Test Voice",
            files=[str(audio_file.resolve())],
            description=None,
        )

    assert len(captured_handles) == 1
    assert all(handle.closed for handle in captured_handles)


def test_voice_clone_missing_file_does_not_call_sdk(tmp_path):
    missing_file = tmp_path / "missing.mp3"

    with patch.object(server, "client") as mock_client:
        assert isinstance(mock_client, MagicMock)

        with pytest.raises(ElevenLabsMcpError):
            server.voice_clone(
                name="Test Voice",
                files=[str(missing_file.resolve())],
                description=None,
            )

        mock_client.voices.ivc.create.assert_not_called()
