from pydantic import BaseModel
from typing import Dict, Optional


class McpVoice(BaseModel):
    id: str
    name: str
    category: str
    fine_tuning_status: Optional[Dict] = None


class ConvAiAgentListItem(BaseModel):
    name: str
    agent_id: str


class ConvaiAgent(BaseModel):
    name: str
    agent_id: str
    system_prompt: str
    voice_id: str | None
    language: str
    llm: str


class McpLanguage(BaseModel):
    language_id: str
    name: str


class McpModel(BaseModel):
    id: str
    name: str
    languages: list[McpLanguage]


class SixtyDbVoice(BaseModel):
    id: str
    name: str
    category: str
    model: Optional[str] = None
    language: Optional[str] = None
    gender: Optional[str] = None
    accent: Optional[str] = None
    description: Optional[str] = None
