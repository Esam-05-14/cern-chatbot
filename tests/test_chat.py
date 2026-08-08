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
    monkeypatch.setenv("CERNGITLAB_CHAT_RAG", "0")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    # Point MCP command at a fake executable path so resolve does not need PATH.
    fake_mcp = tmp_path / "cerngitlab-mcp-stdio"
    fake_mcp.write_text("", encoding="utf-8")
    monkeypatch.setenv("CERNGITLAB_CHAT_MCP_COMMAND", str(fake_mcp))
    # Ignore project .env so local RAG/UI settings cannot hang unit tests.
    return ChatSettings(_env_file=None)


def test_require_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ignore process env and project .env so this asserts the empty-key path.
    monkeypatch.setenv("OPENAI_API_KEY", "")
    settings = ChatSettings(_env_file=None)
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
    monkeypatch.setenv("CERNGITLAB_CHAT_RAG", "0")
    settings = ChatSettings(_env_file=None)

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
    assert "get_file_content" in SYSTEM_PROMPT
    assert "lhcb/Rec" in SYSTEM_PROMPT
    assert "do not invent" in SYSTEM_PROMPT.lower() or "Do NOT invent" in SYSTEM_PROMPT
    assert "retrieve_code" in SYSTEM_PROMPT
    assert "ParticleAdder" not in SYSTEM_PROMPT


def test_warn_if_weak_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("CERNGITLAB_CHAT_MODEL", "openrouter/free")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    weak = ChatSettings(_env_file=None)
    assert weak.warn_if_weak_model() is not None

    monkeypatch.setenv("CERNGITLAB_CHAT_MODEL", "gpt-4.1-mini")
    strong = ChatSettings(_env_file=None)
    assert strong.warn_if_weak_model() is None


def test_extract_tool_logs_from_parts() -> None:
    from types import SimpleNamespace

    from cerngitlab_chat.tool_log import extract_tool_logs

    messages = [
        SimpleNamespace(
            parts=[
                SimpleNamespace(
                    part_kind="tool-call",
                    tool_name="search_code",
                    args={"search_term": "DecayTreeFitter", "project": "lhcb/DaVinci"},
                    tool_call_id="1",
                ),
                SimpleNamespace(
                    part_kind="tool-return",
                    tool_name="search_code",
                    content={"total_results": 2},
                    tool_call_id="1",
                ),
            ]
        )
    ]
    logs = extract_tool_logs(messages)
    assert len(logs) == 2
    assert logs[0].kind == "call"
    assert logs[0].tool == "search_code"
    assert "DecayTreeFitter" in logs[0].detail
    assert logs[1].kind == "result"


@pytest.mark.asyncio
async def test_chat_session_ask_and_clear(chat_settings: ChatSettings) -> None:
    from unittest.mock import AsyncMock

    from cerngitlab_chat.session import ChatSession

    fake_agent = MagicMock()
    fake_agent.__aenter__ = AsyncMock(return_value=fake_agent)
    fake_agent.__aexit__ = AsyncMock(return_value=None)
    first = MagicMock()
    first.output = "answer-1"
    first.all_messages.return_value = ["m1"]
    first.new_messages.return_value = []
    second = MagicMock()
    second.output = "answer-2"
    second.all_messages.return_value = ["m1", "m2"]
    second.new_messages.return_value = []
    fake_agent.run = AsyncMock(side_effect=[first, second])

    with patch("cerngitlab_chat.session.create_agent", return_value=fake_agent):
        session = ChatSession.from_settings(chat_settings)
        async with session:
            turn1 = await session.ask("q1")
            assert turn1.reply == "answer-1"
            turn2 = await session.ask("q2")
            assert turn2.reply == "answer-2"
            assert fake_agent.run.await_count == 2
            # Second call should pass prior history.
            assert fake_agent.run.await_args_list[1].kwargs["message_history"] == ["m1"]
            session.clear_history()
            assert session._history is None


def test_ui_static_and_chat_api(chat_settings: ChatSettings) -> None:
    from unittest.mock import AsyncMock

    from fastapi.testclient import TestClient

    from cerngitlab_chat.session import ChatTurn
    from cerngitlab_chat.tool_log import ToolLogEntry
    from cerngitlab_chat.ui.app import create_ui_app

    fake_session = MagicMock()
    fake_session.__aenter__ = AsyncMock(return_value=fake_session)
    fake_session.__aexit__ = AsyncMock(return_value=None)
    fake_session.ask = AsyncMock(
        return_value=ChatTurn(
            reply="Moore hosts HLT2 lines.",
            tool_logs=[
                ToolLogEntry(
                    kind="call",
                    tool="search_projects",
                    detail='{"query": "Moore"}',
                    tool_call_id="t1",
                )
            ],
        )
    )
    fake_session.clear_history = MagicMock()

    with patch(
        "cerngitlab_chat.ui.app.ChatSession.from_settings", return_value=fake_session
    ):
        app = create_ui_app(chat_settings)
        with TestClient(app) as client:
            home = client.get("/")
            assert home.status_code == 200
            assert b"cerngitlab-chat" in home.content
            assert b"MCP tools required" in home.content

            css = client.get("/static/styles.css")
            assert css.status_code == 200
            assert b"tool-log" in css.content

            status = client.get("/api/status")
            assert status.status_code == 200
            assert status.json()["ready"] is True
            assert status.json()["model"] == "test-model"

            chat = client.post("/api/chat", json={"message": "What is Moore?"})
            assert chat.status_code == 200
            body = chat.json()
            assert "Moore" in body["reply"]
            assert body["used_tools"] is True
            assert body["tool_logs"][0]["tool"] == "search_projects"
            fake_session.ask.assert_awaited_once()

            cleared = client.post("/api/clear")
            assert cleared.status_code == 200
            fake_session.clear_history.assert_called_once()
