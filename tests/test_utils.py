import pytest
from pathlib import Path
import tempfile
from elevenlabs_mcp.utils import (
    ElevenLabsMcpError,
    make_error,
    is_file_writeable,
    make_output_file,
    make_output_path,
    find_similar_filenames,
    try_find_similar_files,
    handle_input_file,
)


def test_make_error():
    with pytest.raises(ElevenLabsMcpError):
        make_error("Test error")


def test_is_file_writeable():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        assert is_file_writeable(temp_path) is True
        assert is_file_writeable(temp_path / "nonexistent.txt") is True


def test_make_output_file():
    tool = "test"
    text = "hello world"
    result = make_output_file(tool, text, "mp3")
    assert result.name.startswith("test_hello")
    assert result.suffix == ".mp3"


def test_make_output_path():
    with tempfile.TemporaryDirectory() as temp_dir:
        result = make_output_path(temp_dir)
        assert result == Path(temp_dir)
        assert result.exists()
        assert result.is_dir()


def test_find_similar_filenames():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test_file.txt"
        similar_file = temp_path / "test_file_2.txt"
        different_file = temp_path / "different.txt"

        test_file.touch()
        similar_file.touch()
        different_file.touch()

        results = find_similar_filenames(str(test_file), temp_path)
        assert len(results) > 0
        assert any(str(similar_file) in str(r[0]) for r in results)


def test_try_find_similar_files():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test_file.mp3"
        similar_file = temp_path / "test_file_2.mp3"
        different_file = temp_path / "different.txt"

        test_file.touch()
        similar_file.touch()
        different_file.touch()

        results = try_find_similar_files(str(test_file), temp_path)
        assert len(results) > 0
        assert any(str(similar_file) in str(r) for r in results)


def test_handle_input_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test.mp3"

        with open(test_file, "wb") as f:
            f.write(b"\xff\xfb\x90\x64\x00")

        result = handle_input_file(str(test_file))
        assert result == test_file

        with pytest.raises(ElevenLabsMcpError):
            handle_input_file(str(temp_path / "nonexistent.mp3"))
