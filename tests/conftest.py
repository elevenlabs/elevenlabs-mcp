import pytest
from pathlib import Path
import tempfile
import os

# server.py reads ELEVENLABS_API_KEY at import time; set a dummy value so
# server-level tests can import it without a real key.
os.environ.setdefault("ELEVENLABS_API_KEY", "test-key")


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_audio_file(temp_dir):
    audio_file = temp_dir / "test.mp3"
    audio_file.touch()
    return audio_file


@pytest.fixture
def sample_video_file(temp_dir):
    video_file = temp_dir / "test.mp4"
    video_file.touch()
    return video_file
