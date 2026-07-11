import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from elevenlabs.types.agent_config import AgentConfig
from elevenlabs.types.conversational_config import ConversationalConfig
from elevenlabs.types.knowledge_base_locator import KnowledgeBaseLocator

os.environ.setdefault("ELEVENLABS_API_KEY", "test-key")

from elevenlabs_mcp import server  # noqa: E402
from elevenlabs_mcp.utils import ElevenLabsMcpError  # noqa: E402


def _agent_with_prompt(knowledge_base=None):
    agent_config = AgentConfig(
        prompt={
            "prompt": "You are helpful.",
            "llm": "gemini-2.0-flash-001",
            "knowledge_base": knowledge_base or [],
        }
    )
    conversation_config = ConversationalConfig(agent=agent_config)
    return SimpleNamespace(
        conversation_config=conversation_config,
    )


def test_add_knowledge_base_url_updates_pydantic_agent_config():
    """SDK returns AgentConfig models; dict-style .get() previously crashed."""
    captured = {}

    def fake_update(agent_id, conversation_config):
        captured["agent_id"] = agent_id
        captured["conversation_config"] = conversation_config
        return SimpleNamespace()

    with patch.object(server, "client") as mock_client:
        mock_client.conversational_ai.knowledge_base.documents.create_from_url.return_value = SimpleNamespace(
            id="doc_url_1"
        )
        mock_client.conversational_ai.agents.get.return_value = _agent_with_prompt()
        mock_client.conversational_ai.agents.update.side_effect = fake_update

        result = server.add_knowledge_base_to_agent(
            agent_id="agent_123",
            knowledge_base_name="Docs",
            url="https://example.com/kb",
        )

    assert "doc_url_1" in result.text
    assert "agent_123" in result.text
    assert captured["agent_id"] == "agent_123"

    updated = captured["conversation_config"]
    assert isinstance(updated, ConversationalConfig)
    assert isinstance(updated.agent, AgentConfig)
    kb = updated.agent.prompt.knowledge_base
    assert len(kb) == 1
    assert isinstance(kb[0], KnowledgeBaseLocator)
    assert kb[0].id == "doc_url_1"
    assert kb[0].type == "url"
    assert kb[0].name == "Docs"


def test_add_knowledge_base_preserves_existing_entries():
    existing = [
        KnowledgeBaseLocator(type="url", name="Old", id="doc_old"),
    ]
    captured = {}

    def fake_update(agent_id, conversation_config):
        captured["conversation_config"] = conversation_config
        return SimpleNamespace()

    with patch.object(server, "client") as mock_client:
        mock_client.conversational_ai.knowledge_base.documents.create_from_url.return_value = SimpleNamespace(
            id="doc_new"
        )
        mock_client.conversational_ai.agents.get.return_value = _agent_with_prompt(
            knowledge_base=existing
        )
        mock_client.conversational_ai.agents.update.side_effect = fake_update

        server.add_knowledge_base_to_agent(
            agent_id="agent_123",
            knowledge_base_name="New",
            url="https://example.com/new",
        )

    kb = captured["conversation_config"].agent.prompt.knowledge_base
    assert [item.id for item in kb] == ["doc_old", "doc_new"]


def test_add_knowledge_base_text_source_uses_file_locator_type():
    captured = {}

    def fake_update(agent_id, conversation_config):
        captured["conversation_config"] = conversation_config
        return SimpleNamespace()

    with patch.object(server, "client") as mock_client:
        mock_client.conversational_ai.knowledge_base.documents.create_from_file.return_value = SimpleNamespace(
            id="doc_text_1"
        )
        mock_client.conversational_ai.agents.get.return_value = _agent_with_prompt()
        mock_client.conversational_ai.agents.update.side_effect = fake_update

        server.add_knowledge_base_to_agent(
            agent_id="agent_123",
            knowledge_base_name="Notes",
            text="hello knowledge",
        )

    locator = captured["conversation_config"].agent.prompt.knowledge_base[0]
    assert locator.type == "file"
    assert locator.id == "doc_text_1"
    mock_client.conversational_ai.knowledge_base.documents.create_from_file.assert_called_once()


def test_add_knowledge_base_file_path_closes_handle(tmp_path):
    kb_file = tmp_path / "notes.txt"
    kb_file.write_text("faq content")
    captured_handles = []

    def fake_create_from_file(name, file):
        captured_handles.append(file)
        assert not file.closed
        assert file.read() == b"faq content"
        file.seek(0)
        return SimpleNamespace(id="doc_file_1")

    with patch.object(server, "client") as mock_client:
        mock_client.conversational_ai.knowledge_base.documents.create_from_file.side_effect = fake_create_from_file
        mock_client.conversational_ai.agents.get.return_value = _agent_with_prompt()
        mock_client.conversational_ai.agents.update.return_value = SimpleNamespace()

        server.add_knowledge_base_to_agent(
            agent_id="agent_123",
            knowledge_base_name="FAQ",
            input_file_path=str(kb_file.resolve()),
        )

    assert len(captured_handles) == 1
    assert captured_handles[0].closed


def test_add_knowledge_base_rejects_missing_prompt():
    agent = SimpleNamespace(
        conversation_config=ConversationalConfig(agent=AgentConfig(prompt=None))
    )

    with patch.object(server, "client") as mock_client:
        mock_client.conversational_ai.knowledge_base.documents.create_from_url.return_value = SimpleNamespace(
            id="doc_1"
        )
        mock_client.conversational_ai.agents.get.return_value = agent

        with pytest.raises(ElevenLabsMcpError, match="prompt configuration"):
            server.add_knowledge_base_to_agent(
                agent_id="agent_123",
                knowledge_base_name="Docs",
                url="https://example.com/kb",
            )

        mock_client.conversational_ai.agents.update.assert_not_called()


def test_add_knowledge_base_requires_exactly_one_source():
    with patch.object(server, "client") as mock_client:
        with pytest.raises(ElevenLabsMcpError, match="exactly one"):
            server.add_knowledge_base_to_agent(
                agent_id="agent_123",
                knowledge_base_name="Docs",
                url="https://example.com/kb",
                text="also text",
            )
        mock_client.conversational_ai.knowledge_base.documents.create_from_url.assert_not_called()
