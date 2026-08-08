"""Configuration for the optional code chatbot CLI."""

from __future__ import annotations

import os
import shutil
import sys
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def default_rag_index_dir() -> Path:
    """Default on-disk location for the FAISS RAG index."""
    return Path.cwd() / ".cache" / "cerngitlab-chat" / "faiss"


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
        description=(
            "Model id passed to the OpenAI-compatible provider. "
            "Prefer a tool-calling capable model (not a random free router) for MCP use."
        ),
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
            "Empty means auto-detect ``cerngitlab-mcp-stdio`` on PATH / venv Scripts."
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
    mcp_retries: int = Field(
        default=1,
        validation_alias="CERNGITLAB_CHAT_MCP_RETRIES",
        description="Extra retries for a full agent turn on timeout/rate-limit style errors",
        ge=0,
        le=5,
    )
    ui_host: str = Field(
        default="127.0.0.1",
        validation_alias="CERNGITLAB_CHAT_UI_HOST",
        description="Bind host for the local web UI",
    )
    ui_port: int = Field(
        default=7860,
        validation_alias="CERNGITLAB_CHAT_UI_PORT",
        description="Bind port for the local web UI",
        ge=1,
        le=65535,
    )
    rag_enabled: bool = Field(
        default=False,
        validation_alias="CERNGITLAB_CHAT_RAG",
        description="Enable local FAISS retrieve_code tool when an index is present",
    )
    rag_index_dir: str = Field(
        default="",
        validation_alias="CERNGITLAB_CHAT_RAG_INDEX_DIR",
        description="Directory containing the FAISS index (empty = .cache/cerngitlab-chat/faiss)",
    )
    embed_model: str = Field(
        default="text-embedding-3-small",
        validation_alias="CERNGITLAB_CHAT_EMBED_MODEL",
        description="OpenAI-compatible embedding model id used by the indexer / RAG",
    )

    @field_validator("openai_base_url", mode="before")
    @classmethod
    def _empty_base_url_to_none(cls, value: object) -> object:
        if value == "":
            return None
        return value

    @field_validator("rag_enabled", mode="before")
    @classmethod
    def _parse_rag_enabled(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return value

    def require_api_key(self) -> str:
        """Return the API key or raise a clear configuration error."""
        if not self.openai_api_key.strip():
            raise ValueError(
                "OPENAI_API_KEY is required. Set it in the environment or a .env file "
                "(works with OpenAI, LiteLLM, Groq, vLLM, and other OpenAI-compatible APIs)."
            )
        return self.openai_api_key

    def warn_if_weak_model(self) -> str | None:
        """Return a warning when the configured model is unsuitable for MCP tool use."""
        model = self.model.strip().lower()
        weak = (
            model in {"openrouter/free", "free"}
            or model.endswith("/free")
            or "openrouter/free" in model
        )
        if not weak:
            return None
        return (
            f"CERNGITLAB_CHAT_MODEL={self.model!r} is a free/router id and is unreliable "
            "for MCP tool calling (rate limits, random upstreams). Pin a concrete "
            "tool-capable model such as gpt-4.1-mini or openai/gpt-4.1-mini."
        )
    def resolve_rag_index_dir(self) -> Path:
        """Return the FAISS index directory path."""
        if self.rag_index_dir.strip():
            return Path(self.rag_index_dir).expanduser().resolve()
        return default_rag_index_dir().resolve()

    def resolve_mcp_command(self) -> str:
        """Resolve the MCP stdio executable path (PATH + active venv Scripts)."""
        if self.mcp_command.strip():
            return self.mcp_command.strip()

        candidates: list[Path] = []
        for name in ("cerngitlab-mcp-stdio", "cerngitlab-mcp"):
            found = shutil.which(name)
            if found:
                return found

        # Prefer .venv-full over .venv: on Windows the default .venv is often
        # locked by Cursor MCP, and mixing both venvs triggers extra AV scans.
        scripts_dirs = [
            Path(sys.executable).resolve().parent,
            Path(sys.prefix) / "Scripts",
            Path(sys.prefix) / "bin",
            Path.cwd() / ".venv-full" / "Scripts",
            Path.cwd() / ".venv-full" / "bin",
            Path.cwd() / ".venv" / "Scripts",
            Path.cwd() / ".venv" / "bin",
        ]
        for scripts in scripts_dirs:
            for name in ("cerngitlab-mcp-stdio.exe", "cerngitlab-mcp-stdio", "cerngitlab-mcp.exe", "cerngitlab-mcp"):
                candidates.append(scripts / name)

        for path in candidates:
            if path.is_file():
                return str(path)

        raise ValueError(
            "Could not find ``cerngitlab-mcp-stdio`` or ``cerngitlab-mcp`` on PATH "
            "or in the active venv Scripts. Install this package "
            "(``uv sync --group chat``) or set CERNGITLAB_CHAT_MCP_COMMAND."
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
