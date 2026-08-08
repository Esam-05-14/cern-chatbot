"""Extract human-readable MCP / tool activity from pydantic-ai messages."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Sequence


ToolLogKind = Literal["call", "result"]


@dataclass(frozen=True)
class ToolLogEntry:
    """One tool call or tool return from an agent turn."""

    kind: ToolLogKind
    tool: str
    detail: str
    tool_call_id: str | None = None


def _truncate(text: str, limit: int = 800) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _format_args(args: object) -> str:
    if args is None:
        return ""
    if isinstance(args, str):
        return _truncate(args)
    try:
        return _truncate(json.dumps(args, ensure_ascii=False, default=str))
    except TypeError:
        return _truncate(str(args))


def _format_content(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return _truncate(content)
    try:
        return _truncate(json.dumps(content, ensure_ascii=False, default=str))
    except TypeError:
        return _truncate(str(content))


def extract_tool_logs(messages: Sequence[Any]) -> list[ToolLogEntry]:
    """Collect tool-call / tool-return parts from a message list (one turn)."""
    logs: list[ToolLogEntry] = []
    for message in messages:
        parts = getattr(message, "parts", None) or ()
        for part in parts:
            part_kind = getattr(part, "part_kind", None)
            tool_name = getattr(part, "tool_name", None)
            if not tool_name:
                continue
            tool_call_id = getattr(part, "tool_call_id", None)
            if part_kind == "tool-call" or type(part).__name__.endswith("ToolCallPart"):
                logs.append(
                    ToolLogEntry(
                        kind="call",
                        tool=str(tool_name),
                        detail=_format_args(getattr(part, "args", None)),
                        tool_call_id=str(tool_call_id) if tool_call_id else None,
                    )
                )
            elif part_kind == "tool-return" or type(part).__name__.endswith(
                "ToolReturnPart"
            ):
                logs.append(
                    ToolLogEntry(
                        kind="result",
                        tool=str(tool_name),
                        detail=_format_content(getattr(part, "content", None)),
                        tool_call_id=str(tool_call_id) if tool_call_id else None,
                    )
                )
    return logs


def tool_logs_as_dicts(logs: Sequence[ToolLogEntry]) -> list[dict[str, str | None]]:
    """Serialize tool logs for JSON API responses."""
    return [
        {
            "kind": entry.kind,
            "tool": entry.tool,
            "detail": entry.detail,
            "tool_call_id": entry.tool_call_id,
        }
        for entry in logs
    ]
