"""Default LHCb stack projects for local RAG indexing."""

from __future__ import annotations

# CERN GitLab paths (group/project) used for Option-2 indexing MVP.
DEFAULT_STACK_PROJECTS: tuple[str, ...] = (
    "lhcb/Moore",
    "lhcb/Allen",
    "lhcb/Gaudi",
    "lhcb/Rec",
    "lhcb/LHCb",
    "lhcb/Gauss",
    "lhcb/DaVinci",
)

GITLAB_BASE = "https://gitlab.cern.ch"
