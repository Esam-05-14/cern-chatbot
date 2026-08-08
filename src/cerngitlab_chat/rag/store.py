"""FAISS-backed local code index for cerngitlab-chat RAG."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from cerngitlab_chat.rag.chunking import CodeChunk


META_NAME = "chunks.jsonl"
INDEX_NAME = "index.faiss"
INFO_NAME = "index_info.json"


@dataclass
class RetrievedChunk:
    """A scored retrieval hit."""

    project: str
    path: str
    ref: str
    start_line: int
    end_line: int
    symbol: str
    text: str
    score: float


class FaissCodeStore:
    """Persist and query a FAISS index with parallel chunk metadata."""

    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir)
        self._index: Any = None
        self._chunks: list[dict[str, Any]] = []
        self._dim: int | None = None

    @property
    def size(self) -> int:
        return len(self._chunks)

    def is_ready(self) -> bool:
        return (
            (self.index_dir / INDEX_NAME).is_file()
            and (self.index_dir / META_NAME).is_file()
        )

    def build(self, chunks: Sequence[CodeChunk], vectors: Sequence[Sequence[float]]) -> None:
        """Create a new index from chunks + embedding vectors."""
        import faiss

        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        if not chunks:
            raise ValueError("No chunks to index")

        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("vectors must be a 2D array")
        faiss.normalize_L2(matrix)
        dim = matrix.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(matrix)

        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_dir / INDEX_NAME))
        with (self.index_dir / META_NAME).open("w", encoding="utf-8") as fh:
            for chunk in chunks:
                fh.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
        info = {"dim": dim, "count": len(chunks)}
        (self.index_dir / INFO_NAME).write_text(
            json.dumps(info, indent=2), encoding="utf-8"
        )
        self._index = index
        self._chunks = [asdict(c) for c in chunks]
        self._dim = dim

    def load(self) -> None:
        """Load index + metadata from disk."""
        import faiss

        if not self.is_ready():
            raise FileNotFoundError(
                f"RAG index not found under {self.index_dir}. "
                "Run ``cerngitlab-chat-index`` first."
            )
        self._index = faiss.read_index(str(self.index_dir / INDEX_NAME))
        self._chunks = []
        with (self.index_dir / META_NAME).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self._chunks.append(json.loads(line))
        info_path = self.index_dir / INFO_NAME
        if info_path.is_file():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            self._dim = int(info.get("dim") or 0) or None

    def search(self, query_vector: Sequence[float], *, top_k: int = 8) -> list[RetrievedChunk]:
        """Return top_k chunks by inner-product similarity (normalized vectors)."""
        import faiss

        if self._index is None:
            self.load()
        assert self._index is not None

        vec = np.asarray([list(query_vector)], dtype=np.float32)
        faiss.normalize_L2(vec)
        k = min(max(1, top_k), max(1, len(self._chunks)))
        scores, indices = self._index.search(vec, k)
        hits: list[RetrievedChunk] = []
        for score, idx in zip(scores[0].tolist(), indices[0].tolist(), strict=False):
            if idx < 0 or idx >= len(self._chunks):
                continue
            meta = self._chunks[idx]
            hits.append(
                RetrievedChunk(
                    project=str(meta.get("project", "")),
                    path=str(meta.get("path", "")),
                    ref=str(meta.get("ref", "")),
                    start_line=int(meta.get("start_line") or 0),
                    end_line=int(meta.get("end_line") or 0),
                    symbol=str(meta.get("symbol") or ""),
                    text=str(meta.get("text") or ""),
                    score=float(score),
                )
            )
        return hits
