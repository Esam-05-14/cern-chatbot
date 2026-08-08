"""Tests for RAG chunking / FAISS store / backend wiring (no live GitLab)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cerngitlab_chat.config import ChatSettings
from cerngitlab_chat.prompts import SYSTEM_PROMPT
from cerngitlab_chat.rag.chunking import chunk_file
from cerngitlab_chat.rag.store import FaissCodeStore


@pytest.fixture
def chat_settings(tmp_path, monkeypatch: pytest.MonkeyPatch) -> ChatSettings:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("CERNGITLAB_CHAT_MODEL", "test-model")
    monkeypatch.setenv("CERNGITLAB_CHAT_RAG", "0")
    fake_mcp = tmp_path / "cerngitlab-mcp-stdio"
    fake_mcp.write_text("", encoding="utf-8")
    monkeypatch.setenv("CERNGITLAB_CHAT_MCP_COMMAND", str(fake_mcp))
    monkeypatch.setenv("CERNGITLAB_CHAT_RAG_INDEX_DIR", str(tmp_path / "faiss"))
    return ChatSettings(_env_file=None)


def test_hybrid_prompt_mentions_retrieve_code_first() -> None:
    assert "retrieve_code" in SYSTEM_PROMPT
    assert "do not know" in SYSTEM_PROMPT.lower()
    assert "get_file_content" in SYSTEM_PROMPT
    assert "ALWAYS use CERN GitLab MCP tools" not in SYSTEM_PROMPT
    assert "ParticleAdder" not in SYSTEM_PROMPT


def test_line_window_chunking(tmp_path: Path) -> None:
    source = tmp_path / "sample.py"
    body = "\n".join(f"line_{i} = {i}" for i in range(120))
    source.write_text(body, encoding="utf-8")
    chunks = chunk_file(source, project="lhcb/Demo", rel_path="sample.py", ref="master")
    assert chunks
    assert chunks[0].project == "lhcb/Demo"
    assert chunks[0].path == "sample.py"
    assert chunks[0].start_line >= 1


def test_faiss_store_roundtrip(tmp_path: Path) -> None:
    faiss = pytest.importorskip("faiss")
    _ = faiss
    from cerngitlab_chat.rag.chunking import CodeChunk

    chunks = [
        CodeChunk(
            project="lhcb/Demo",
            path="a.py",
            ref="master",
            start_line=1,
            end_line=5,
            symbol="foo",
            text="def foo():\n    return 1\n",
        ),
        CodeChunk(
            project="lhcb/Demo",
            path="b.py",
            ref="master",
            start_line=1,
            end_line=5,
            symbol="bar",
            text="def bar():\n    return 2\n",
        ),
    ]
    # Simple orthogonal-ish vectors
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    store = FaissCodeStore(tmp_path / "idx")
    store.build(chunks, vectors)
    store2 = FaissCodeStore(tmp_path / "idx")
    store2.load()
    hits = store2.search([1.0, 0.0, 0.0], top_k=1)
    assert hits
    assert hits[0].symbol == "foo"


def test_default_backend_adds_rag_when_enabled(
    chat_settings: ChatSettings, tmp_path: Path
) -> None:
    from cerngitlab_chat.backends import default_tool_backend
    from cerngitlab_chat.rag.chunking import CodeChunk
    from cerngitlab_chat.rag.store import FaissCodeStore

    pytest.importorskip("faiss")

    index_dir = tmp_path / "faiss"
    chunks = [
        CodeChunk(
            project="lhcb/Demo",
            path="a.py",
            ref="master",
            start_line=1,
            end_line=2,
            symbol="foo",
            text="def foo():\n    return 1\n",
        )
    ]
    FaissCodeStore(index_dir).build(chunks, [[1.0, 0.0]])

    settings = chat_settings.model_copy(
        update={"rag_enabled": True, "rag_index_dir": str(index_dir)}
    )
    fake_toolset = MagicMock(name="mcp")
    with (
        patch(
            "cerngitlab_chat.backends.McpStdioBackend.build_toolsets",
            return_value=[fake_toolset],
        ),
        patch("cerngitlab_chat.rag.backend.Embedder"),
    ):
        backend = default_tool_backend(settings)
    assert len(backend.backends) == 2
    assert backend.backends[1].name == "code-rag"


def test_rag_retrieve_code_json(chat_settings: ChatSettings, tmp_path: Path) -> None:
    import json

    pytest.importorskip("faiss")
    from cerngitlab_chat.rag.backend import RagToolBackend
    from cerngitlab_chat.rag.chunking import CodeChunk

    index_dir = tmp_path / "faiss"
    chunks = [
        CodeChunk(
            project="lhcb/DaVinci",
            path="tutorial.py",
            ref="master",
            start_line=1,
            end_line=10,
            symbol="main",
            text="from DecayTreeFitter import DecayTreeFitter\n",
        )
    ]
    FaissCodeStore(index_dir).build(chunks, [[0.2, 0.8]])
    settings = chat_settings.model_copy(update={"rag_index_dir": str(index_dir)})
    rag = RagToolBackend(settings=settings)

    with patch.object(rag, "_ensure_loaded") as ensure:
        store = FaissCodeStore(index_dir)
        store.load()
        embedder = MagicMock()
        embedder.embed_query.return_value = [0.2, 0.8]
        ensure.return_value = (store, embedder)
        payload = json.loads(rag.retrieve_code("DecayTreeFitter", top_k=1))

    assert payload["hits"]
    assert payload["hits"][0]["project"] == "lhcb/DaVinci"
