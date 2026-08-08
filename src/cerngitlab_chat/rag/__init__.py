"""Local FAISS RAG helpers for cerngitlab-chat (optional ``rag`` dependency group)."""

from cerngitlab_chat.rag.backend import RagToolBackend
from cerngitlab_chat.rag.projects import DEFAULT_STACK_PROJECTS

__all__ = ["DEFAULT_STACK_PROJECTS", "RagToolBackend"]
