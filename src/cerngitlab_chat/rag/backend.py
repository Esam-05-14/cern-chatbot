"""RAG tool backend exposing retrieve_code via pydantic-ai FunctionToolset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from cerngitlab_chat.config import ChatSettings
from cerngitlab_chat.rag.embed import Embedder
from cerngitlab_chat.rag.store import FaissCodeStore

if TYPE_CHECKING:
    from pydantic_ai.toolsets import AbstractToolset


@dataclass
class RagToolBackend:
    """Local FAISS retrieve_code tool for hybrid RAG + MCP answering."""

    settings: ChatSettings
    name: str = "code-rag"
    _store: FaissCodeStore | None = None
    _embedder: Embedder | None = None

    def index_dir(self) -> Path:
        return self.settings.resolve_rag_index_dir()

    def available(self) -> bool:
        return FaissCodeStore(self.index_dir()).is_ready()

    def _ensure_loaded(self) -> tuple[FaissCodeStore, Embedder]:
        if self._store is None:
            store = FaissCodeStore(self.index_dir())
            store.load()
            self._store = store
        if self._embedder is None:
            self._embedder = Embedder(self.settings)
        return self._store, self._embedder

    def retrieve_code(self, query: str, top_k: int = 8) -> str:
        """Retrieve relevant LHCb code chunks from the local FAISS index.

        Use this first for fast recall over indexed stack projects, then confirm
        important paths with live CERN GitLab MCP tools (search_code /
        get_file_content).
        """
        store, embedder = self._ensure_loaded()
        vector = embedder.embed_query(query)
        hits = store.search(vector, top_k=top_k)
        payload = [
            {
                "project": hit.project,
                "path": hit.path,
                "ref": hit.ref,
                "start_line": hit.start_line,
                "end_line": hit.end_line,
                "symbol": hit.symbol,
                "score": round(hit.score, 4),
                "text": hit.text[:2000],
            }
            for hit in hits
        ]
        return json.dumps({"query": query, "hits": payload}, ensure_ascii=False)

    def build_toolsets(self) -> Sequence[AbstractToolset]:
        from pydantic_ai.toolsets import FunctionToolset

        toolset = FunctionToolset(tools=[self.retrieve_code], id=self.name)
        return [toolset]
