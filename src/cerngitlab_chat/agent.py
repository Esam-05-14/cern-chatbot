"""pydantic-ai agent factory for MCP-backed LHCb code Q&A."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from cerngitlab_chat.backends import ToolBackend, default_tool_backend
from cerngitlab_chat.config import ChatSettings, get_chat_settings
from cerngitlab_chat.prompts import SYSTEM_PROMPT

if TYPE_CHECKING:
    from pydantic_ai import Agent
    from pydantic_ai.models import Model


def build_model(settings: ChatSettings) -> Model:
    """Create an OpenAI-compatible pydantic-ai model from chat settings."""
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    api_key = settings.require_api_key()
    provider_kwargs: dict[str, object] = {"api_key": api_key}
    if settings.openai_base_url:
        provider_kwargs["base_url"] = settings.openai_base_url

    provider = OpenAIProvider(**provider_kwargs)
    return OpenAIChatModel(settings.model, provider=provider)


def create_agent(
    settings: ChatSettings | None = None,
    *,
    tool_backend: ToolBackend | None = None,
    system_prompt: str | None = None,
) -> Agent[None, str]:
    """Create the code Q&A agent with MCP (and optional extra) toolsets.

    Args:
        settings: Chat settings; defaults to env / ``.env``.
        tool_backend: Tool source; defaults to CERN GitLab MCP over stdio.
            Pass a :class:`~cerngitlab_chat.backends.CompositeToolBackend` with
            extra backends to attach Phase-2 RAG tools without changing the CLI.
        system_prompt: Override the default LHCb Q&A system prompt.
    """
    from pydantic_ai import Agent

    settings = settings or get_chat_settings()
    backend = tool_backend or default_tool_backend(settings)
    toolsets: Sequence[object] = backend.build_toolsets()

    return Agent(
        build_model(settings),
        system_prompt=system_prompt or SYSTEM_PROMPT,
        toolsets=toolsets,
    )
