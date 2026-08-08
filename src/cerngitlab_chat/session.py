"""Shared chat session used by the CLI and web UI."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from cerngitlab_chat.agent import create_agent
from cerngitlab_chat.config import ChatSettings, get_chat_settings
from cerngitlab_chat.project_paths import annotate_user_question
from cerngitlab_chat.tool_log import ToolLogEntry, extract_tool_logs


def usage_limits(settings: ChatSettings) -> Any:
    """Build pydantic-ai usage limits from chat settings."""
    from pydantic_ai.usage import UsageLimits

    return UsageLimits(request_limit=settings.max_tool_rounds + 1)


def _is_retryable_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    # Daily free-tier quotas will not recover with short sleeps — do not retry.
    hard = (
        "free-models-per-day",
        "free_tier_daily",
        "insufficient credits",
        "requires more credits",
        "quota exceeded",
        "payment required",
        "status_code: 402",
        "status_code:402",
    )
    if any(h in text for h in hard):
        return False
    needles = (
        "timeout",
        "timed out",
        "rate limit",
        "too many requests",
        "429",
        "503",
        "temporarily unavailable",
        "connection reset",
        "connect error",
    )
    return any(n in text for n in needles)


def is_hard_quota_error(exc: BaseException | str) -> bool:
    """True when the provider daily/credit quota is exhausted (abort eval)."""
    text = str(exc).lower()
    return any(
        h in text
        for h in (
            "free-models-per-day",
            "free_tier_daily",
            "insufficient credits",
            "requires more credits",
            "quota exceeded",
            "payment required",
            "status_code: 402",
            "status_code:402",
        )
    )


@dataclass(frozen=True)
class ChatTurn:
    """One user question → assistant reply, including MCP tool activity."""

    reply: str
    tool_logs: list[ToolLogEntry] = field(default_factory=list)

    @property
    def used_tools(self) -> bool:
        return any(entry.kind == "call" for entry in self.tool_logs)


@dataclass
class ChatSession:
    """Long-lived MCP-backed agent session with conversation history.

    The MCP subprocess stays open for the lifetime of the session so multi-turn
    chat (CLI REPL or web UI) does not re-handshake on every message.
    """

    settings: ChatSettings
    _agent: Any = field(default=None, init=False, repr=False)
    _history: list[Any] | None = field(default=None, init=False, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _started: bool = field(default=False, init=False, repr=False)

    @classmethod
    def from_settings(cls, settings: ChatSettings | None = None) -> ChatSession:
        """Create a session from settings (defaults to env / ``.env``)."""
        return cls(settings=settings or get_chat_settings())

    async def __aenter__(self) -> ChatSession:
        self.settings.require_api_key()
        warning = self.settings.warn_if_weak_model()
        if warning:
            import sys

            print(f"WARNING: {warning}", file=sys.stderr)
        self.settings.resolve_mcp_command()
        self._agent = create_agent(self.settings)
        await self._agent.__aenter__()
        self._started = True
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._agent is not None and self._started:
            await self._agent.__aexit__(*exc)
        self._started = False
        self._agent = None
        self._history = None

    async def ask(self, question: str) -> ChatTurn:
        """Send a user message and return the reply plus tool logs for this turn."""
        text = annotate_user_question(question.strip())
        if not text:
            raise ValueError("Question must not be empty.")
        if not self._started or self._agent is None:
            raise RuntimeError("ChatSession is not started; use ``async with``.")

        attempts = 1 + max(0, self.settings.mcp_retries)
        last_error: BaseException | None = None

        async with self._lock:
            for attempt in range(attempts):
                try:
                    result = await self._agent.run(
                        text,
                        message_history=self._history,
                        usage_limits=usage_limits(self.settings),
                    )
                    self._history = result.all_messages()
                    new_messages = result.new_messages()
                    tool_logs = extract_tool_logs(new_messages)
                    return ChatTurn(reply=str(result.output), tool_logs=tool_logs)
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    if attempt + 1 >= attempts or not _is_retryable_error(exc):
                        raise
                    await asyncio.sleep(1.5 * (attempt + 1))

        assert last_error is not None
        raise last_error

    def clear_history(self) -> None:
        """Drop multi-turn context while keeping the MCP agent alive."""
        self._history = None
