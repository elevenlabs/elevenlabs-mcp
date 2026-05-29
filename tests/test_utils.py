import pytest
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock
from elevenlabs_mcp.utils import (
    ElevenLabsMcpError,
    make_error,
    is_file_writeable,
    make_output_file,
    make_output_path,
    find_similar_filenames,
    try_find_similar_files,
    handle_input_file,
    looks_like_unsubstituted_template,
    parse_location,
    resolve_resource_path,
)
from elevenlabs_mcp.server import simulate_conversation


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


def test_make_output_path_none_output_directory():
    """Test with None as output_directory, should use base_path"""
    with tempfile.TemporaryDirectory() as temp_dir:
        result = make_output_path(None, temp_dir)
        assert result == Path(temp_dir)
        assert result.exists()
        assert result.is_dir()


def test_make_output_path_none_output_directory_none_base_path():
    """Test with both None, should default to ~/Desktop"""
    with tempfile.TemporaryDirectory() as temp_home:
        mock_home = Path(temp_home)
        with patch("elevenlabs_mcp.utils.Path.home", return_value=mock_home):
            result = make_output_path(None, None)
            assert result == mock_home / "Desktop"
            assert result.exists()
            assert result.is_dir()


def test_make_output_path_relative_no_base_path():
    """Test edge case: relative path with no base_path, should use ~/Desktop as base"""
    with tempfile.TemporaryDirectory() as temp_home:
        mock_home = Path(temp_home)
        # Create Desktop directory so the parent exists
        desktop_dir = mock_home / "Desktop"
        desktop_dir.mkdir()

        with patch("elevenlabs_mcp.utils.Path.home", return_value=mock_home):
            relative_subdir = "test_subdir"
            expected = desktop_dir / relative_subdir

            result = make_output_path(relative_subdir, None)
            assert result == expected
            assert result.exists()
            assert result.is_dir()


def test_make_output_path_absolute_path():
    """Test with absolute output_directory, should ignore base_path"""
    with tempfile.TemporaryDirectory() as temp_dir:
        absolute_path = Path(temp_dir) / "absolute_test"
        result = make_output_path(str(absolute_path), "/some/ignored/base")
        assert result == absolute_path
        assert result.exists()
        assert result.is_dir()


def test_make_output_path_relative_with_base():
    """Test with relative output_directory and base_path"""
    with tempfile.TemporaryDirectory() as temp_dir:
        relative_subdir = "subdir"
        result = make_output_path(relative_subdir, temp_dir)
        assert result == Path(temp_dir) / relative_subdir
        assert result.exists()
        assert result.is_dir()


def test_looks_like_unsubstituted_template_detects_dollar_brace():
    assert looks_like_unsubstituted_template("${user_config.output_dir}")
    assert looks_like_unsubstituted_template("  ${user_config.output_dir}  ")


def test_looks_like_unsubstituted_template_detects_double_brace():
    assert looks_like_unsubstituted_template("{{user_config.output_dir}}")


def test_looks_like_unsubstituted_template_rejects_real_paths():
    assert not looks_like_unsubstituted_template("/tmp/output")
    assert not looks_like_unsubstituted_template("/tmp/$weird/dir")
    assert not looks_like_unsubstituted_template("~/Desktop")
    assert not looks_like_unsubstituted_template("")
    assert not looks_like_unsubstituted_template(None)
    # A path that merely contains a brace expression in the middle isn't a placeholder.
    assert not looks_like_unsubstituted_template("/tmp/${x}/y")


def test_make_output_path_unsubstituted_base_path_falls_back_to_default(capsys):
    """Cowork plugin wrapper bug: literal '${user_config.output_dir}' must not crash."""
    with tempfile.TemporaryDirectory() as temp_home:
        mock_home = Path(temp_home)
        desktop = mock_home / "Desktop"
        desktop.mkdir()
        with patch("elevenlabs_mcp.utils.Path.home", return_value=mock_home):
            result = make_output_path(None, "${user_config.output_dir}")
            assert result == desktop
            assert result.exists()
        stderr = capsys.readouterr().err
        assert "unsubstituted" in stderr


def test_make_output_path_unsubstituted_double_brace_base_path(capsys):
    with tempfile.TemporaryDirectory() as temp_home:
        mock_home = Path(temp_home)
        desktop = mock_home / "Desktop"
        desktop.mkdir()
        with patch("elevenlabs_mcp.utils.Path.home", return_value=mock_home):
            result = make_output_path(None, "{{user_config.output_dir}}")
            assert result == desktop
            assert result.exists()


def test_make_output_path_unsubstituted_output_directory_is_ignored(capsys):
    with tempfile.TemporaryDirectory() as temp_dir:
        result = make_output_path("${user_config.subdir}", temp_dir)
        # output_directory dropped → falls back to using base_path alone
        assert result == Path(temp_dir)
        assert result.exists()
        stderr = capsys.readouterr().err
        assert "unsubstituted" in stderr


def test_resolve_resource_path_absolute_inside_base_dir(tmp_path):
    target = tmp_path / "sub" / "file.mp3"
    target.parent.mkdir()
    target.write_bytes(b"x")
    result = resolve_resource_path(str(target), tmp_path)
    assert result == target.resolve()


def test_resolve_resource_path_absolute_outside_base_dir(tmp_path):
    outside = tmp_path.parent / "escape.txt"
    with pytest.raises(ElevenLabsMcpError):
        resolve_resource_path(str(outside), tmp_path)


def test_resolve_resource_path_relative_traversal(tmp_path):
    with pytest.raises(ElevenLabsMcpError):
        resolve_resource_path("../escape.txt", tmp_path)


def test_resolve_resource_path_relative_normal(tmp_path):
    target = tmp_path / "ok.mp3"
    target.write_bytes(b"x")
    result = resolve_resource_path("ok.mp3", tmp_path)
    assert result == target.resolve()


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

def test_simulate_conversation_bad_criteria_returns_error():
    """Missing fields in evaluation criteria should return an error without calling API."""
    with patch("elevenlabs_mcp.server.client") as mock_client:
        result = simulate_conversation(
            agent_id="agent_abc",
            simulated_user_prompt="Be difficult.",
            extra_evaluation_criteria=[{"id": "check"}],  # missing name + goal
        )
        assert "missing" in result.text.lower()
        mock_client.conversational_ai.agents.simulate_conversation.assert_not_called()


def test_simulate_conversation_formats_transcript():
    """Conversation turns should appear correctly in output."""
    with patch("elevenlabs_mcp.server.client") as mock_client:
        user_turn = MagicMock()
        user_turn.role = "user"
        user_turn.message = "I need help with billing."
        user_turn.tool_calls = []

        agent_turn = MagicMock()
        agent_turn.role = "agent"
        agent_turn.message = "I can help you with that."
        agent_turn.tool_calls = []

        analysis = MagicMock()
        analysis.transcript_summary = "Billing issue resolved."
        analysis.call_successful = "success"
        analysis.evaluation_criteria_results = {}

        mock_response = MagicMock()
        mock_response.simulated_conversation = [user_turn, agent_turn]
        mock_response.analysis = analysis
        mock_client.conversational_ai.agents.simulate_conversation.return_value = mock_response

        result = simulate_conversation(
            agent_id="agent_abc",
            simulated_user_prompt="You are a customer with a billing question.",
        )
        assert "I need help with billing." in result.text
        assert "I can help you with that." in result.text
        assert "Billing issue resolved." in result.text
        assert "✅" in result.text


def test_simulate_conversation_handles_empty_response():
    """Empty conversation history should not crash."""
    with patch("elevenlabs_mcp.server.client") as mock_client:
        mock_response = MagicMock()
        mock_response.simulated_conversation = []
        mock_response.analysis = None
        mock_client.conversational_ai.agents.simulate_conversation.return_value = mock_response

        result = simulate_conversation(
            agent_id="agent_abc",
            simulated_user_prompt="Do nothing.",
        )
        assert result.text is not None


def test_parse_location_shorthands():
    """Test that 'eu' and 'in' shorthands resolve to the same URLs as their full forms."""
    assert parse_location("eu") == "https://api.eu.residency.elevenlabs.io"
    assert parse_location("in") == "https://api.in.residency.elevenlabs.io"
    assert parse_location("eu") == parse_location("eu-residency")
    assert parse_location("in") == parse_location("in-residency")


def test_parse_location_existing_values():
    """Existing residency values still work."""
    assert parse_location("us") == "https://api.elevenlabs.io"
    assert parse_location("global") == "https://api.elevenlabs.io"
    assert parse_location("eu-residency") == "https://api.eu.residency.elevenlabs.io"
    assert parse_location("in-residency") == "https://api.in.residency.elevenlabs.io"


def test_parse_location_none_and_empty():
    """None and empty strings default to US."""
    assert parse_location(None) == "https://api.elevenlabs.io"
    assert parse_location("") == "https://api.elevenlabs.io"
    assert parse_location("   ") == "https://api.elevenlabs.io"


def test_parse_location_case_insensitive():
    """Shorthands are case-insensitive."""
    assert parse_location("EU") == "https://api.eu.residency.elevenlabs.io"
    assert parse_location("IN") == "https://api.in.residency.elevenlabs.io"
    assert parse_location("  Eu  ") == "https://api.eu.residency.elevenlabs.io"


def test_parse_location_invalid():
    """Invalid values raise ValueError."""
    with pytest.raises(ValueError):
        parse_location("invalid")
