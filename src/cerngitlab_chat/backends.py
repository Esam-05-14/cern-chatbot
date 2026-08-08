"""Pluggable tool backends for the chatbot (MCP now; RAG later).

Phase 1 wires CERN GitLab MCP over stdio.
Phase 2 can register additional :class:`~pydantic_ai.toolsets.AbstractToolset`
instances (for example a ``retrieve_code`` FunctionToolset over FAISS) without
changing the CLI entry point.
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
        # Prefer the dedicated stdio entry point when the resolved command is
        # the auto-mode ``cerngitlab-mcp`` binary so HTTP mode is not selected.
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
    """Combine multiple backends (MCP + future RAG) into one toolset list."""

    backends: list[ToolBackend] = field(default_factory=list)
    name: str = "composite"

    def add(self, backend: ToolBackend) -> None:
        """Register an additional tool backend (e.g. future RAG)."""
        self.backends.append(backend)

    def build_toolsets(self) -> Sequence[AbstractToolset]:
        toolsets: list[AbstractToolset] = []
        for backend in self.backends:
            toolsets.extend(backend.build_toolsets())
        return toolsets


def default_tool_backend(settings: ChatSettings | None = None) -> CompositeToolBackend:
    """Build the default Phase-1 backend stack (MCP only).

    Phase 2: call ``backend.add(RagToolBackend(...))`` before creating the agent,
    or pass a custom :class:`CompositeToolBackend` into :func:`create_agent`.
    """
    from cerngitlab_chat.config import get_chat_settings

    settings = settings or get_chat_settings()
    return CompositeToolBackend(backends=[McpStdioBackend(settings=settings)])
