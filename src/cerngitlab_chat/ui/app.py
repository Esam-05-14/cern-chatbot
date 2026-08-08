"""Local web UI for the MCP-backed code chatbot."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from cerngitlab_chat.config import ChatSettings, get_chat_settings
from cerngitlab_chat.session import ChatSession
from cerngitlab_chat.tool_log import tool_logs_as_dicts

STATIC_DIR = Path(__file__).resolve().parent / "static"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20000)


class ToolLogItem(BaseModel):
    kind: str
    tool: str
    detail: str = ""
    tool_call_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    model: str
    tool_logs: list[ToolLogItem] = Field(default_factory=list)
    used_tools: bool = False


class StatusResponse(BaseModel):
    ok: bool
    model: str
    ready: bool
    detail: str | None = None


class _UiState:
    """Mutable app state for the single local chat session."""

    def __init__(self) -> None:
        self.settings: ChatSettings | None = None
        self.session: ChatSession | None = None


def create_ui_app(settings: ChatSettings | None = None) -> FastAPI:
    """Build the FastAPI app that serves the chat UI and JSON API."""
    app_settings = settings or get_chat_settings()
    state = _UiState()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        state.settings = app_settings
        session = ChatSession.from_settings(app_settings)
        await session.__aenter__()
        state.session = session
        try:
            yield
        finally:
            if state.session is not None:
                await state.session.__aexit__(None, None, None)
                state.session = None

    app = FastAPI(
        title="cerngitlab-chat",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )

    @app.get("/api/status", response_model=StatusResponse)
    async def status() -> StatusResponse:
        ready = state.session is not None and state.settings is not None
        return StatusResponse(
            ok=True,
            model=state.settings.model if state.settings else app_settings.model,
            ready=ready,
            detail=None if ready else "Session starting",
        )

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(body: ChatRequest) -> ChatResponse:
        if state.session is None or state.settings is None:
            raise HTTPException(status_code=503, detail="Chat session is not ready yet.")
        try:
            turn = await state.session.ask(body.message)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return ChatResponse(
            reply=turn.reply,
            model=state.settings.model,
            tool_logs=[ToolLogItem(**item) for item in tool_logs_as_dicts(turn.tool_logs)],
            used_tools=turn.used_tools,
        )

    @app.post("/api/clear")
    async def clear() -> dict[str, bool]:
        if state.session is None:
            raise HTTPException(status_code=503, detail="Chat session is not ready yet.")
        state.session.clear_history()
        return {"ok": True}

    @app.get("/")
    async def index() -> FileResponse:
        index_path = STATIC_DIR / "index.html"
        if not index_path.is_file():
            raise HTTPException(status_code=500, detail="UI assets missing.")
        return FileResponse(index_path)

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app


def run_ui(
    settings: ChatSettings | None = None,
    *,
    host: str | None = None,
    port: int | None = None,
) -> None:
    """Run the chat UI with uvicorn (blocking)."""
    import uvicorn

    settings = settings or get_chat_settings()
    bind_host = host or settings.ui_host
    bind_port = port if port is not None else settings.ui_port
    app = create_ui_app(settings)
    print(f"cerngitlab-chat UI -> http://{bind_host}:{bind_port}/")
    uvicorn.run(app, host=bind_host, port=bind_port, log_level="info")
