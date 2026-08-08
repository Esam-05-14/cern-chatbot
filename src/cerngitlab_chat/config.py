"""Configuration for the optional code chatbot CLI."""

from __future__ import annotations

import os
import shutil
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChatSettings(BaseSettings):
    """Chatbot settings loaded from environment variables.

    OpenAI-compatible credentials use the standard ``OPENAI_*`` names so LiteLLM,
    Groq, vLLM, and OpenAI all work the same way.
    Chat-specific knobs use the ``CERNGITLAB_CHAT_`` prefix.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(
        default="",
        validation_alias="OPENAI_API_KEY",
        description="API key for an OpenAI-compatible endpoint",
    )
    openai_base_url: str | None = Field(
        default=None,
        validation_alias="OPENAI_BASE_URL",
        description="Optional OpenAI-compatible base URL (LiteLLM, Groq, vLLM, …)",
    )
    model: str = Field(
        default="gpt-4.1-mini",
        validation_alias="CERNGITLAB_CHAT_MODEL",
        description="Model id passed to the OpenAI-compatible provider",
    )
    max_tool_rounds: int = Field(
        default=5,
        validation_alias="CERNGITLAB_CHAT_MAX_TOOL_ROUNDS",
        description="Maximum model request rounds per user question (tool-call budget)",
        ge=1,
        le=50,
    )
    mcp_command: str = Field(
        default="",
        validation_alias="CERNGITLAB_CHAT_MCP_COMMAND",
        description=(
            "Executable used to spawn the CERN GitLab MCP server over stdio. "
            "Empty means auto-detect ``cerngitlab-mcp-stdio`` on PATH."
        ),
    )
    mcp_timeout: float = Field(
        default=30.0,
        validation_alias="CERNGITLAB_CHAT_MCP_TIMEOUT",
        description="Seconds to wait for the MCP handshake",
        gt=0,
    )
    mcp_read_timeout: float = Field(
        default=300.0,
        validation_alias="CERNGITLAB_CHAT_MCP_READ_TIMEOUT",
        description="Seconds to wait for individual MCP tool responses",
        gt=0,
    )

    @field_validator("openai_base_url", mode="before")
    @classmethod
    def _empty_base_url_to_none(cls, value: object) -> object:
        if value == "":
            return None
        return value

    def require_api_key(self) -> str:
        """Return the API key or raise a clear configuration error."""
        if not self.openai_api_key.strip():
            raise ValueError(
                "OPENAI_API_KEY is required. Set it in the environment or a .env file "
                "(works with OpenAI, LiteLLM, Groq, vLLM, and other OpenAI-compatible APIs)."
            )
        return self.openai_api_key

    def resolve_mcp_command(self) -> str:
        """Resolve the MCP stdio executable path."""
        if self.mcp_command.strip():
            return self.mcp_command.strip()
        found = shutil.which("cerngitlab-mcp-stdio")
        if found:
            return found
        found = shutil.which("cerngitlab-mcp")
        if found:
            return found
        raise ValueError(
            "Could not find ``cerngitlab-mcp-stdio`` or ``cerngitlab-mcp`` on PATH. "
            "Install this package (``uv sync`` / ``pip install -e .``) or set "
            "CERNGITLAB_CHAT_MCP_COMMAND to the executable."
        )


def mcp_child_env() -> dict[str, str]:
    """Environment for the MCP subprocess.

    Passes through the parent environment (including ``CERNGITLAB_*``) so the
    child server sees the same GitLab token and settings as the CLI user.
    """
    return dict(os.environ)


@lru_cache
def get_chat_settings() -> ChatSettings:
    """Return cached chatbot settings."""
    return ChatSettings()
