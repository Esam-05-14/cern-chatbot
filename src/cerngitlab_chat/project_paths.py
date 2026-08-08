"""Canonical CERN GitLab LHCb project paths (tooling helpers, not prompt priming)."""

from __future__ import annotations

# Preferred GitLab paths for stack / analysis tooling.
CANONICAL_PROJECTS: dict[str, str] = {
    "moore": "lhcb/Moore",
    "allen": "lhcb/Allen",
    "gaudi": "lhcb/Gaudi",
    "rec": "lhcb/Rec",
    "lhcb": "lhcb/LHCb",
    "gauss": "lhcb/Gauss",
    "davinci": "lhcb/DaVinci",
    "detector": "lhcb/Detector",
    "boole": "lhcb/Boole",
    "brunel": "lhcb/Brunel",
}

# Common wrong spellings / casing the model has used in tool calls.
PROJECT_ALIASES: dict[str, str] = {
    "lhcb/reco": "lhcb/Rec",
    "lhcb/rec": "lhcb/Rec",
    "lhcb/res": "lhcb/Rec",
    "lhcb/moore": "lhcb/Moore",
    "lhcb/allen": "lhcb/Allen",
    "lhcb/gaudi": "lhcb/Gaudi",
    "lhcb/gauss": "lhcb/Gauss",
    "lhcb/davinci": "lhcb/DaVinci",
    "lhcb/detector": "lhcb/Detector",
    "lhcb/lhcb": "lhcb/LHCb",
}

# Explicit misspellings / wrong casing that must never be sent to GitLab tools.
_KNOWN_BAD_SPELLINGS = (
    "lhcb/reco",
    "lhcb/res",
    "lhcb/moore",
    "lhcb/allen",
    "lhcb/gaudi",
    "lhcb/gauss",
    "lhcb/davinci",
    "lhcb/detector",
    "lhcb/lhcb",
    "lhcb/rec",  # wrong casing vs lhcb/Rec
)


def canonicalize_project_path(project: str) -> str:
    """Return a canonical ``group/name`` path when known."""
    key = project.strip().strip("/")
    lower = key.lower()
    if lower in PROJECT_ALIASES:
        return PROJECT_ALIASES[lower]
    # Bare names
    bare = lower.split("/")[-1]
    if bare in CANONICAL_PROJECTS and "/" not in lower:
        return CANONICAL_PROJECTS[bare]
    if lower.startswith("lhcb/") and bare in CANONICAL_PROJECTS:
        return CANONICAL_PROJECTS[bare]
    return key


def find_bad_project_paths(text: str) -> list[str]:
    """List non-canonical lhcb/* spellings that should be corrected.

    Canonical paths like ``lhcb/Moore`` / ``lhcb/Rec`` are never flagged.
    """
    found: list[str] = []
    blob = text or ""
    for bad in _KNOWN_BAD_SPELLINGS:
        # Exact lowercase / known-bad token matches only (avoids flagging lhcb/Moore).
        start = 0
        while True:
            idx = blob.find(bad, start)
            if idx < 0:
                break
            # Require non-word boundaries so we don't match mid-token.
            before_ok = idx == 0 or not blob[idx - 1].isalnum()
            end = idx + len(bad)
            after_ok = end >= len(blob) or not (blob[end].isalnum() or blob[end] == "_")
            if before_ok and after_ok and bad not in found:
                found.append(bad)
            start = end
    return found


def annotate_user_question(question: str) -> str:
    """Normalize the user question for the agent.

    Does not inject domain or routing hints: experts expect the model to deduce
    answers from tools (or say it does not know), not from primed text.
    """
    return question.strip()
