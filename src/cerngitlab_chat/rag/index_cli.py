"""Indexer CLI: clone/read LHCb stack sources → chunk → embed → FAISS."""

from __future__ import annotations

import subprocess
from pathlib import Path

import click

from cerngitlab_chat.config import ChatSettings, default_rag_index_dir, get_chat_settings
from cerngitlab_chat.rag.chunking import CodeChunk, chunk_file, iter_source_files
from cerngitlab_chat.rag.embed import Embedder
from cerngitlab_chat.rag.projects import DEFAULT_STACK_PROJECTS, GITLAB_BASE
from cerngitlab_chat.rag.store import FaissCodeStore


def _project_name(project: str) -> str:
    return project.rstrip("/").split("/")[-1]


def _ensure_clone(
    project: str,
    repos_dir: Path,
    *,
    ref: str,
    depth: int,
) -> Path:
    name = _project_name(project)
    dest = repos_dir / name
    url = f"{GITLAB_BASE}/{project}.git"
    if dest.is_dir() and (dest / ".git").exists():
        click.echo(f"Using existing clone {dest}")
        return dest
    repos_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "git",
        "clone",
        "--depth",
        str(depth),
        "--branch",
        ref,
        "--single-branch",
        url,
        str(dest),
    ]
    click.echo(f"Cloning {project} @ {ref} …")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        # Fallback without branch (default HEAD)
        click.echo(f"Branch {ref} unavailable; cloning default HEAD for {project}")
        subprocess.run(
            ["git", "clone", "--depth", str(depth), url, str(dest)],
            check=True,
        )
    return dest


def _collect_chunks(
    *,
    projects: list[str],
    repos_dir: Path,
    ref: str,
    clone: bool,
    depth: int,
    max_files_per_project: int | None,
) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []
    for project in projects:
        name = _project_name(project)
        root = repos_dir / name
        if clone or not root.is_dir():
            if not clone and not root.is_dir():
                raise click.ClickException(
                    f"Missing local checkout {root}. Pass --clone or place repos under {repos_dir}."
                )
            if clone:
                root = _ensure_clone(project, repos_dir, ref=ref, depth=depth)
        files = iter_source_files(root)
        if max_files_per_project is not None:
            files = files[: max_files_per_project]
        click.echo(f"Chunking {project}: {len(files)} files")
        for file_path in files:
            rel = str(file_path.relative_to(root)).replace("\\", "/")
            chunks.extend(
                chunk_file(file_path, project=project, rel_path=rel, ref=ref)
            )
    return chunks


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--repos-dir",
    type=click.Path(path_type=Path),
    default=Path(".cache/cerngitlab-chat/repos"),
    show_default=True,
    help="Directory containing (or receiving) project checkouts",
)
@click.option(
    "--index-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="FAISS output directory (default: .cache/cerngitlab-chat/faiss)",
)
@click.option(
    "--project",
    "projects",
    multiple=True,
    help="GitLab project path (repeatable). Default: core LHCb stack set.",
)
@click.option("--ref", default="master", show_default=True, help="Git ref/branch to record and clone")
@click.option("--clone/--no-clone", default=False, help="Shallow-clone missing projects from gitlab.cern.ch")
@click.option("--depth", default=1, show_default=True, help="Shallow clone depth")
@click.option(
    "--max-files",
    type=int,
    default=None,
    help="Optional per-project file cap (useful for smoke tests)",
)
@click.option(
    "--max-chunks",
    type=int,
    default=None,
    help="Optional total chunk cap after collection (smoke tests)",
)
def main(
    repos_dir: Path,
    index_dir: Path | None,
    projects: tuple[str, ...],
    ref: str,
    clone: bool,
    depth: int,
    max_files: int | None,
    max_chunks: int | None,
) -> None:
    """Build a local FAISS code index for cerngitlab-chat RAG."""
    get_chat_settings.cache_clear()
    settings = get_chat_settings()
    settings.require_api_key()

    project_list = list(projects) if projects else list(DEFAULT_STACK_PROJECTS)
    out_dir = index_dir or settings.resolve_rag_index_dir()
    if index_dir is None and not settings.rag_index_dir.strip():
        out_dir = default_rag_index_dir()

    click.echo(f"Projects: {', '.join(project_list)}")
    click.echo(f"Repos dir: {repos_dir}")
    click.echo(f"Index dir: {out_dir}")
    click.echo(f"Embed model: {settings.embed_model}")

    chunks = _collect_chunks(
        projects=project_list,
        repos_dir=repos_dir,
        ref=ref,
        clone=clone,
        depth=depth,
        max_files_per_project=max_files,
    )
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
    if not chunks:
        raise click.ClickException("No chunks produced; check repos-dir / --clone / --project.")

    click.echo(f"Embedding {len(chunks)} chunks …")
    embedder = Embedder(settings)
    vectors = embedder.embed_texts([c.text for c in chunks])
    store = FaissCodeStore(out_dir)
    store.build(chunks, vectors)
    click.echo(f"Wrote FAISS index with {store.size} chunks → {out_dir}")
    click.echo("Enable RAG with CERNGITLAB_CHAT_RAG=1 (and restart cerngitlab-chat).")


if __name__ == "__main__":
    main()
