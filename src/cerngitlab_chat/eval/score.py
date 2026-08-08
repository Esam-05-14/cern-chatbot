"""Heuristic quality scoring for chatbot eval transcripts (no live LLM judge)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from cerngitlab_chat.project_paths import find_bad_project_paths

_PATH_CITE_RE = re.compile(
    r"(?:lhcb/[\w.\-]+)|(?:[\w.\-]+/(?:[\w.\-]+/)+\S+\.(?:py|cpp|h|hpp|md))",
    re.IGNORECASE,
)


@dataclass
class ScoreResult:
    """Numeric score plus structured issue list."""

    score: float
    max_score: float = 100.0
    issues: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.score >= 70.0 and not any(
            i.startswith("FAIL:") for i in self.issues
        )


def _tool_names(tool_logs: Sequence[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for entry in tool_logs:
        if entry.get("kind") == "call" and entry.get("tool"):
            names.append(str(entry["tool"]))
    return names


def _tool_details(tool_logs: Sequence[dict[str, Any]]) -> str:
    return " ".join(str(e.get("detail") or "") for e in tool_logs)


# Questions that may be answered from system knowledge without MCP calls.
_NO_TOOL_IDS = frozenset(
    {
        "hybrid_rag_mcp_role",
        "stack_projects_list",
        "gitlab_search_hygiene",
        "recoconf_not_a_project",
    }
)


def score_answer(
    *,
    question_spec: dict[str, Any],
    reply: str,
    tool_logs: Sequence[dict[str, Any]] | None = None,
    used_tools: bool | None = None,
    error: str | None = None,
) -> ScoreResult:
    """Score one answer using topic-specific heuristics.

    Designed so *good* grounded transcripts pass and *bad* ones (wrong projects,
    no file reads, invented CHI2DOF stories) fail.
    """
    tool_logs = list(tool_logs or [])
    reply = reply or ""
    issues: list[str] = []
    checks: dict[str, bool] = {}
    score = 100.0

    if error:
        issues.append(f"FAIL: run error: {error}")
        return ScoreResult(score=0.0, issues=issues, checks={"error": True})

    tools = _tool_names(tool_logs)
    details = _tool_details(tool_logs)
    blob_for_paths = f"{details}\n{reply}"
    if used_tools is None:
        used_tools = bool(tools)

    qid = str(question_spec.get("id") or "")
    checks["used_tools"] = bool(used_tools)
    if qid not in _NO_TOOL_IDS and not used_tools:
        score -= 40
        issues.append("FAIL: no tools were used")

    if question_spec.get("forbid_bad_projects", True):
        bad_found = find_bad_project_paths(blob_for_paths)
        checks["no_bad_projects"] = not bad_found
        if bad_found:
            score -= 35
            issues.append(f"FAIL: bad project path(s): {bad_found}")

    expected_projects = question_spec.get("expect_search_projects_any") or []
    if expected_projects:
        ok = any(p in details for p in expected_projects)
        checks["searched_expected_project"] = ok
        if not ok:
            score -= 20
            issues.append(f"FAIL: expected a search in one of {expected_projects}")

    if question_spec.get("require_get_file_content"):
        ok = "get_file_content" in tools
        checks["get_file_content"] = ok
        if not ok:
            score -= 25
            issues.append("FAIL: missing get_file_content after code search")

    expect_tools = question_spec.get("expect_tools_any") or []
    if expect_tools:
        ok = any(t in tools for t in expect_tools)
        checks["expect_tools"] = ok
        if not ok:
            score -= 15
            issues.append(f"missing preferred tools among {expect_tools}")

    must_any = question_spec.get("must_include_any") or []
    if must_any:
        ok = any(s.lower() in reply.lower() for s in must_any)
        checks["must_include"] = ok
        if not ok:
            score -= 20
            issues.append(f"FAIL: reply missing any of {must_any}")

    cite_projects = question_spec.get("must_cite_project_any") or []
    if cite_projects:
        blob = reply + " " + details
        ok = any(p in blob for p in cite_projects)
        checks["cite_project"] = ok
        if not ok:
            score -= 15
            issues.append(f"missing citation to one of {cite_projects}")

    has_path = bool(_PATH_CITE_RE.search(reply))
    checks["path_citation"] = has_path
    if question_spec.get("require_get_file_content") or cite_projects:
        if not has_path:
            score -= 10
            issues.append("weak path citation in reply")

    for phrase in question_spec.get("forbid_phrases_any") or []:
        if phrase.lower() in reply.lower():
            score -= 25
            issues.append(f"FAIL: forbidden phrase present: {phrase!r}")
            checks["forbid_phrases"] = False
            break
    else:
        if question_spec.get("forbid_phrases_any"):
            checks["forbid_phrases"] = True

    if len(reply.strip()) < 40:
        score -= 20
        issues.append("reply too short")

    score = max(0.0, min(100.0, score))
    if any(i.startswith("FAIL:") for i in issues) and score >= 70:
        score = 69.0
    return ScoreResult(score=score, issues=issues, checks=checks)
