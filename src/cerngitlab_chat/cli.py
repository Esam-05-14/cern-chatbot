"""CLI entry point for the optional CERN GitLab / LHCb code chatbot."""

from __future__ import annotations

import asyncio
import sys

import click

from cerngitlab_chat.config import get_chat_settings
from cerngitlab_chat.session import ChatSession


async def _run_once(question: str, session: ChatSession) -> str:
    async with session:
        turn = await session.ask(question)
        return turn.reply


async def _run_repl(session: ChatSession) -> int:
    click.echo("cerngitlab-chat — LHCb / CERN GitLab code Q&A (MCP-backed)")
    click.echo("Type a question, or /quit to exit.")
    click.echo("")

    try:
        async with session:
            while True:
                try:
                    question = click.prompt("you", prompt_suffix="> ").strip()
                except (EOFError, click.Abort):
                    click.echo("")
                    break

                if not question:
                    continue
                if question.lower() in {"/quit", "/exit", "quit", "exit"}:
                    break
                if question.lower() in {"/clear", "clear"}:
                    session.clear_history()
                    click.echo("(history cleared)")
                    continue

                try:
                    turn = await session.ask(question)
                except Exception as exc:  # noqa: BLE001 — surface tool/LLM errors to the user
                    click.echo(f"error: {exc}", err=True)
                    continue

                if turn.tool_logs:
                    click.echo("tools:")
                    for entry in turn.tool_logs:
                        prefix = "→" if entry.kind == "call" else "←"
                        detail = f" {entry.detail}" if entry.detail else ""
                        click.echo(f"  {prefix} {entry.tool}{detail}")
                click.echo("")
                click.echo(f"assistant> {turn.reply}")
                click.echo("")
    except Exception as exc:  # noqa: BLE001
        click.echo(f"Failed to start chatbot: {exc}", err=True)
        return 1

    return 0


def _validate_settings(model: str | None):
    get_chat_settings.cache_clear()
    settings = get_chat_settings()
    if model:
        settings = settings.model_copy(update={"model": model})
    try:
        settings.require_api_key()
        settings.resolve_mcp_command()
    except ValueError as exc:
        click.echo(str(exc), err=True)
        raise SystemExit(2) from exc
    return settings


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--question",
    "-q",
    default=None,
    help="Ask a single question and exit (non-interactive).",
)
@click.option(
    "--model",
    default=None,
    help="Override CERNGITLAB_CHAT_MODEL for this run.",
)
@click.option(
    "--ui",
    "launch_ui",
    is_flag=True,
    default=False,
    help="Launch the local web chat UI instead of the terminal REPL.",
)
@click.option(
    "--host",
    default=None,
    help="UI bind host (default from CERNGITLAB_CHAT_UI_HOST or 127.0.0.1).",
)
@click.option(
    "--port",
    type=int,
    default=None,
    help="UI bind port (default from CERNGITLAB_CHAT_UI_PORT or 7860).",
)
def main(
    question: str | None,
    model: str | None,
    launch_ui: bool,
    host: str | None,
    port: int | None,
) -> None:
    """Interactive LHCb code Q&A chatbot using CERN GitLab MCP tools.

    Requires OPENAI_API_KEY (OpenAI-compatible). Optional OPENAI_BASE_URL for
    LiteLLM / Groq / vLLM. GitLab access uses CERNGITLAB_TOKEN and related
    CERNGITLAB_* settings for the MCP subprocess.

    Use ``--ui`` for a local browser chat interface.
    """
    settings = _validate_settings(model)

    if launch_ui:
        if question:
            click.echo("Ignoring --question when --ui is set.", err=True)
        from cerngitlab_chat.ui.app import run_ui

        run_ui(settings=settings, host=host, port=port)
        return

    session = ChatSession.from_settings(settings)

    if question:
        try:
            answer = asyncio.run(_run_once(question, session))
        except Exception as exc:  # noqa: BLE001
            click.echo(f"error: {exc}", err=True)
            raise SystemExit(1) from exc
        click.echo(answer)
        return

    if not sys.stdin.isatty():
        click.echo(
            "No TTY detected. Pass --question/-q or --ui for non-interactive use.",
            err=True,
        )
        raise SystemExit(2)

    raise SystemExit(asyncio.run(_run_repl(session)))


if __name__ == "__main__":
    main()
