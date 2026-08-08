"""Pluggable tool backends for the chatbot (MCP + optional RAG).

Phase 1 wires CERN GitLab MCP over stdio.
Phase 2 can register :class:`~cerngitlab_chat.rag.backend.RagToolBackend`
(``retrieve_code`` over FAISS) via :class:`CompositeToolBackend`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:
    from pydantic_ai.toolsets import AbstractToolset

    from cerngitlab_chat.config import ChatSettings


@runtime_checkable
class ToolBackend(Protocol):
    """Source of agent toolsets that can be opened for a chat session."""

    name: str

    def build_toolsets(self) -> Sequence[AbstractToolset]:
        """Return toolsets to attach to the pydantic-ai Agent."""


@dataclass
class McpStdioBackend:
    """Spawn ``cerngitlab-mcp`` over stdio and expose its tools to the agent."""

    settings: ChatSettings
    name: str = "cerngitlab-mcp"

    def build_toolsets(self) -> Sequence[AbstractToolset]:
        from fastmcp.client.transports import StdioTransport
        from pydantic_ai.mcp import MCPToolset

        from cerngitlab_chat.config import mcp_child_env

        command = self.settings.resolve_mcp_command()
        args: list[str] = []
        if Path(command).stem == "cerngitlab-mcp":
            args = ["--mode", "stdio"]

        transport = StdioTransport(
            command=command,
            args=args,
            env=mcp_child_env(),
            keep_alive=False,
        )
        toolset = MCPToolset(
            transport,
            id=self.name,
            init_timeout=self.settings.mcp_timeout,
            read_timeout=self.settings.mcp_read_timeout,
        )
        return [toolset]


@dataclass
class CompositeToolBackend:
    """Combine multiple backends (MCP + RAG) into one toolset list."""

    backends: list[ToolBackend] = field(default_factory=list)
    name: str = "composite"

    def add(self, backend: ToolBackend) -> None:
        """Register an additional tool backend (e.g. RAG)."""
        self.backends.append(backend)

    def build_toolsets(self) -> Sequence[AbstractToolset]:
        toolsets: list[AbstractToolset] = []
        for backend in self.backends:
            toolsets.extend(backend.build_toolsets())
        return toolsets


def default_tool_backend(settings: ChatSettings | None = None) -> CompositeToolBackend:
    """Build the default backend stack (MCP, plus RAG when enabled and indexed).

    Set ``CERNGITLAB_CHAT_RAG=1`` and build an index with ``cerngitlab-chat-index``.
    """
    from cerngitlab_chat.config import get_chat_settings

    settings = settings or get_chat_settings()
    composite = CompositeToolBackend(backends=[McpStdioBackend(settings=settings)])

    if settings.rag_enabled:
        try:
            from cerngitlab_chat.rag.backend import RagToolBackend
        except ImportError as exc:  # pragma: no cover - missing optional deps
            raise ImportError(
                "CERNGITLAB_CHAT_RAG is enabled but RAG dependencies are missing. "
                "Install with ``uv sync --group rag`` or ``pip install -e '.[rag]'``."
            ) from exc
        rag = RagToolBackend(settings=settings)
        if not rag.available():
            raise FileNotFoundError(
                f"CERNGITLAB_CHAT_RAG is enabled but no index was found under "
                f"{settings.resolve_rag_index_dir()}. Run ``cerngitlab-chat-index`` first."
            )
        composite.add(rag)

    return composite
