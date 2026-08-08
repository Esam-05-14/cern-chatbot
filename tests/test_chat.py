"""Unit tests for cerngitlab-chat wiring (no live GitLab or LLM calls)."""

from __future__ import annotations

from typing import Sequence
from unittest.mock import MagicMock, patch

import pytest

from cerngitlab_chat.backends import (
    CompositeToolBackend,
    McpStdioBackend,
    default_tool_backend,
)
from cerngitlab_chat.config import ChatSettings
from cerngitlab_chat.prompts import SYSTEM_PROMPT


@pytest.fixture
def chat_settings(tmp_path, monkeypatch: pytest.MonkeyPatch) -> ChatSettings:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("CERNGITLAB_CHAT_MODEL", "test-model")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    # Point MCP command at a fake executable path so resolve does not need PATH.
    fake_mcp = tmp_path / "cerngitlab-mcp-stdio"
    fake_mcp.write_text("", encoding="utf-8")
    monkeypatch.setenv("CERNGITLAB_CHAT_MCP_COMMAND", str(fake_mcp))
    return ChatSettings()


def test_require_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = ChatSettings(openai_api_key="")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        settings.require_api_key()


def test_default_tool_backend_is_mcp_only(chat_settings: ChatSettings) -> None:
    backend = default_tool_backend(chat_settings)
    assert isinstance(backend, CompositeToolBackend)
    assert len(backend.backends) == 1
    assert isinstance(backend.backends[0], McpStdioBackend)
    assert backend.backends[0].name == "cerngitlab-mcp"


def test_composite_backend_add_extra() -> None:
    class FakeBackend:
        name = "rag"

        def build_toolsets(self) -> Sequence[object]:
            return ["rag-toolset"]

    composite = CompositeToolBackend(backends=[])
    composite.add(FakeBackend())  # type: ignore[arg-type]
    assert composite.build_toolsets() == ["rag-toolset"]


def test_mcp_stdio_backend_builds_server(chat_settings: ChatSettings) -> None:
    fake_toolset = MagicMock(name="MCPToolset")
    fake_transport = MagicMock(name="StdioTransport")
    with (
        patch(
            "fastmcp.client.transports.StdioTransport", return_value=fake_transport
        ) as transport_ctor,
        patch("pydantic_ai.mcp.MCPToolset", return_value=fake_toolset) as toolset_ctor,
    ):
        toolsets = McpStdioBackend(settings=chat_settings).build_toolsets()

    assert toolsets == [fake_toolset]
    transport_ctor.assert_called_once()
    assert transport_ctor.call_args.kwargs["command"] == chat_settings.mcp_command
    assert transport_ctor.call_args.kwargs["env"] is not None
    toolset_ctor.assert_called_once()
    assert toolset_ctor.call_args.args[0] is fake_transport
    assert toolset_ctor.call_args.kwargs["init_timeout"] == chat_settings.mcp_timeout
    assert toolset_ctor.call_args.kwargs["read_timeout"] == chat_settings.mcp_read_timeout


def test_mcp_auto_mode_binary_gets_stdio_flag(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_mcp = tmp_path / "cerngitlab-mcp.exe"
    fake_mcp.write_text("", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("CERNGITLAB_CHAT_MCP_COMMAND", str(fake_mcp))
    settings = ChatSettings()

    fake_toolset = MagicMock()
    fake_transport = MagicMock()
    with (
        patch(
            "fastmcp.client.transports.StdioTransport", return_value=fake_transport
        ) as transport_ctor,
        patch("pydantic_ai.mcp.MCPToolset", return_value=fake_toolset),
    ):
        McpStdioBackend(settings=settings).build_toolsets()

    assert transport_ctor.call_args.kwargs["command"] == str(fake_mcp)
    assert transport_ctor.call_args.kwargs["args"] == ["--mode", "stdio"]


def test_create_agent_wires_model_and_toolsets(chat_settings: ChatSettings) -> None:
    fake_model = MagicMock(name="model")
    fake_agent = MagicMock(name="agent")
    fake_toolset = MagicMock(name="toolset")

    class Backend:
        name = "test"

        def build_toolsets(self) -> Sequence[object]:
            return [fake_toolset]

    with (
        patch("cerngitlab_chat.agent.build_model", return_value=fake_model),
        patch("pydantic_ai.Agent", return_value=fake_agent) as agent_ctor,
    ):
        from cerngitlab_chat.agent import create_agent

        agent = create_agent(chat_settings, tool_backend=Backend())  # type: ignore[arg-type]

    assert agent is fake_agent
    agent_ctor.assert_called_once()
    call_kwargs = agent_ctor.call_args.kwargs
    assert call_kwargs["system_prompt"] == SYSTEM_PROMPT
    assert call_kwargs["toolsets"] == [fake_toolset]
    assert agent_ctor.call_args.args[0] is fake_model


def test_build_model_passes_base_url(chat_settings: ChatSettings) -> None:
    settings = chat_settings.model_copy(
        update={"openai_base_url": "https://litellm.example/v1"}
    )
    fake_provider = MagicMock()
    fake_model = MagicMock()
    with (
        patch(
            "pydantic_ai.providers.openai.OpenAIProvider", return_value=fake_provider
        ) as provider_ctor,
        patch(
            "pydantic_ai.models.openai.OpenAIChatModel", return_value=fake_model
        ) as model_ctor,
    ):
        from cerngitlab_chat.agent import build_model

        model = build_model(settings)

    assert model is fake_model
    provider_ctor.assert_called_once_with(
        api_key="test-key", base_url="https://litellm.example/v1"
    )
    model_ctor.assert_called_once_with("test-model", provider=fake_provider)


def test_system_prompt_mentions_stack_and_tools() -> None:
    assert "Moore" in SYSTEM_PROMPT
    assert "search_lhcb_stack" in SYSTEM_PROMPT
    assert "retrieve_code" in SYSTEM_PROMPT
