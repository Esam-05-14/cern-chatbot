"""OpenAI-compatible embedding client for RAG indexing and query."""

from __future__ import annotations

from typing import Sequence

from cerngitlab_chat.config import ChatSettings


class Embedder:
    """Thin wrapper around the OpenAI embeddings API (LiteLLM/OpenRouter compatible)."""

    def __init__(self, settings: ChatSettings):
        from openai import OpenAI

        kwargs: dict[str, object] = {"api_key": settings.require_api_key()}
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        self._client = OpenAI(**kwargs)
        self.model = settings.embed_model

    def embed_texts(self, texts: Sequence[str], *, batch_size: int = 32) -> list[list[float]]:
        """Embed texts in batches; returns one vector per input string."""
        vectors: list[list[float]] = []
        batch: list[str] = []
        for text in texts:
            cleaned = text.replace("\x00", "")[:8000]
            batch.append(cleaned if cleaned.strip() else " ")
            if len(batch) >= batch_size:
                vectors.extend(self._embed_batch(batch))
                batch = []
        if batch:
            vectors.extend(self._embed_batch(batch))
        return vectors

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.model, input=batch)
        # API may not preserve order in exotic gateways; sort by index when present.
        data = sorted(response.data, key=lambda row: getattr(row, "index", 0))
        return [list(row.embedding) for row in data]
