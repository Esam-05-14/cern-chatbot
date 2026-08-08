"""AST-aware and fallback code chunking for RAG indexing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CodeChunk:
    """One indexable code snippet with source metadata."""

    project: str
    path: str
    ref: str
    start_line: int
    end_line: int
    symbol: str
    text: str


_PY_EXTENSIONS = {".py"}
_CPP_EXTENSIONS = {".cpp", ".cc", ".cxx", ".h", ".hpp", ".hh", ".hxx", ".inl"}
_SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    "build",
    "InstallArea",
    "node_modules",
    ".tox",
    ".mypy_cache",
}


def _line_window_chunks(
    text: str,
    *,
    project: str,
    path: str,
    ref: str,
    window: int = 80,
    overlap: int = 16,
) -> list[CodeChunk]:
    lines = text.splitlines()
    if not lines:
        return []
    chunks: list[CodeChunk] = []
    step = max(1, window - overlap)
    for start in range(0, len(lines), step):
        end = min(len(lines), start + window)
        body = "\n".join(lines[start:end]).strip()
        if len(body) < 40:
            if end >= len(lines):
                break
            continue
        chunks.append(
            CodeChunk(
                project=project,
                path=path,
                ref=ref,
                start_line=start + 1,
                end_line=end,
                symbol="",
                text=body,
            )
        )
        if end >= len(lines):
            break
    return chunks


def _chunk_python_tree_sitter(
    source: bytes,
    *,
    project: str,
    path: str,
    ref: str,
) -> list[CodeChunk] | None:
    try:
        import tree_sitter_python as tspython
        from tree_sitter import Language, Parser
    except ImportError:
        return None

    language = Language(tspython.language())
    parser = Parser(language)
    tree = parser.parse(source)
    text = source.decode("utf-8", errors="replace")
    lines = text.splitlines()
    chunks: list[CodeChunk] = []
    interesting = {"function_definition", "class_definition"}
    stack = [tree.root_node]
    seen_spans: set[tuple[int, int]] = set()
    while stack:
        node = stack.pop()
        stack.extend(reversed(list(node.children)))
        if node.type not in interesting:
            continue
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        span = (start_line, end_line)
        if span in seen_spans:
            continue
        seen_spans.add(span)
        symbol = ""
        for child in node.children:
            if child.type == "identifier":
                symbol = text[child.start_byte : child.end_byte]
                break
        body = "\n".join(lines[start_line - 1 : end_line]).strip()
        if len(body) < 40:
            continue
        chunks.append(
            CodeChunk(
                project=project,
                path=path,
                ref=ref,
                start_line=start_line,
                end_line=end_line,
                symbol=symbol,
                text=body[:8000],
            )
        )
    return chunks or None


def _chunk_cpp_tree_sitter(
    source: bytes,
    *,
    project: str,
    path: str,
    ref: str,
) -> list[CodeChunk] | None:
    try:
        import tree_sitter_cpp as tscpp
        from tree_sitter import Language, Parser
    except ImportError:
        return None

    language = Language(tscpp.language())
    parser = Parser(language)
    tree = parser.parse(source)
    root = tree.root_node
    text = source.decode("utf-8", errors="replace")
    lines = text.splitlines()
    chunks: list[CodeChunk] = []

    interesting = {
        "function_definition",
        "class_specifier",
        "struct_specifier",
        "declaration",
    }
    stack = [root]
    while stack:
        node = stack.pop()
        stack.extend(reversed(list(node.children)))
        if node.type not in interesting:
            continue
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        if end_line - start_line < 2:
            continue
        body = "\n".join(lines[start_line - 1 : end_line]).strip()
        if len(body) < 60:
            continue
        chunks.append(
            CodeChunk(
                project=project,
                path=path,
                ref=ref,
                start_line=start_line,
                end_line=end_line,
                symbol=node.type,
                text=body[:8000],
            )
        )
    return chunks or None


def chunk_file(
    file_path: Path,
    *,
    project: str,
    rel_path: str,
    ref: str,
) -> list[CodeChunk]:
    """Chunk a single source file using AST when available, else line windows."""
    try:
        source = file_path.read_bytes()
    except OSError:
        return []
    if len(source) > 1_500_000:
        return []

    suffix = file_path.suffix.lower()
    text = source.decode("utf-8", errors="replace")

    if suffix in _PY_EXTENSIONS:
        ast_chunks = _chunk_python_tree_sitter(
            source, project=project, path=rel_path, ref=ref
        )
        if ast_chunks:
            return ast_chunks
    elif suffix in _CPP_EXTENSIONS:
        ast_chunks = _chunk_cpp_tree_sitter(
            source, project=project, path=rel_path, ref=ref
        )
        if ast_chunks:
            return ast_chunks

    return _line_window_chunks(text, project=project, path=rel_path, ref=ref)


def iter_source_files(root: Path) -> list[Path]:
    """List indexable source files under a project checkout."""
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in _SKIP_DIR_NAMES for part in path.parts):
            continue
        if path.suffix.lower() in _PY_EXTENSIONS | _CPP_EXTENSIONS:
            files.append(path)
    return files
