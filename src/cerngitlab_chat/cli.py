"""CLI entry point for the optional CERN GitLab / LHCb code chatbot."""

from __future__ import annotations

import asyncio
import sys
from typing import Any

import click

from cerngitlab_chat.config import ChatSettings, get_chat_settings


def _usage_limits(settings: ChatSettings) -> Any:
    from pydantic_ai.usage import UsageLimits

    # request_limit counts model calls; tool rounds roughly match Team A's ~5 turns.
    return UsageLimits(request_limit=settings.max_tool_rounds + 1)


async def _run_once(question: str, settings: ChatSettings) -> str:
    from cerngitlab_chat.agent import create_agent

    agent = create_agent(settings)
    async with agent:
        result = await agent.run(question, usage_limits=_usage_limits(settings))
    return str(result.output)


async def _run_repl(settings: ChatSettings) -> int:
    from cerngitlab_chat.agent import create_agent

    click.echo("cerngitlab-chat — LHCb / CERN GitLab code Q&A (MCP-backed)")
    click.echo("Type a question, or /quit to exit.")
    click.echo("")

    agent = create_agent(settings)
    message_history: list[Any] | None = None

    try:
        async with agent:
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

                try:
                    result = await agent.run(
                        question,
                        message_history=message_history,
                        usage_limits=_usage_limits(settings),
                    )
                except Exception as exc:  # noqa: BLE001 — surface tool/LLM errors to the user
                    click.echo(f"error: {exc}", err=True)
                    continue

                message_history = result.all_messages()
                click.echo("")
                click.echo(f"assistant> {result.output}")
                click.echo("")
    except Exception as exc:  # noqa: BLE001
        click.echo(f"Failed to start chatbot: {exc}", err=True)
        return 1

    return 0


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
def main(question: str | None, model: str | None) -> None:
    """Interactive LHCb code Q&A chatbot using CERN GitLab MCP tools.

    Requires OPENAI_API_KEY (OpenAI-compatible). Optional OPENAI_BASE_URL for
    LiteLLM / Groq / vLLM. GitLab access uses CERNGITLAB_TOKEN and related
    CERNGITLAB_* settings for the MCP subprocess.
    """
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

    if question:
        try:
            answer = asyncio.run(_run_once(question, settings))
        except Exception as exc:  # noqa: BLE001
            click.echo(f"error: {exc}", err=True)
            raise SystemExit(1) from exc
        click.echo(answer)
        return

    if not sys.stdin.isatty():
        click.echo(
            "No TTY detected. Pass --question/-q for non-interactive use.",
            err=True,
        )
        raise SystemExit(2)

    raise SystemExit(asyncio.run(_run_repl(settings)))


if __name__ == "__main__":
    main()
