"""Tests for chatbot quality heuristics, prompts, and project-path routing."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cerngitlab_chat.eval.questions import EVAL_QUESTIONS
from cerngitlab_chat.eval.score import score_answer
from cerngitlab_chat.project_paths import (
    CANONICAL_PROJECTS,
    annotate_user_question,
    canonicalize_project_path,
    find_bad_project_paths,
)
from cerngitlab_chat.prompts import SYSTEM_PROMPT


def _spec(qid: str) -> dict[str, Any]:
    for item in EVAL_QUESTIONS:
        if item["id"] == qid:
            return item
    raise KeyError(qid)


def test_system_prompt_enforces_tool_discipline() -> None:
    assert "get_file_content" in SYSTEM_PROMPT
    assert "lhcb/reco" in SYSTEM_PROMPT
    assert "do not invent" in SYSTEM_PROMPT.lower()
    assert "retrieve_code" in SYSTEM_PROMPT
    assert "do not know" in SYSTEM_PROMPT.lower()
    # Must not prime domain conclusions (experts deduce from tools).
    assert "ParticleAdder" not in SYSTEM_PROMPT
    assert "CHI2DOF" not in SYSTEM_PROMPT
    assert "DecayTreeFitter" not in SYSTEM_PROMPT
    assert "RecoConf" not in SYSTEM_PROMPT
    assert "Domain reminders" not in SYSTEM_PROMPT
    assert "a|b" in SYSTEM_PROMPT or "regex" in SYSTEM_PROMPT.lower()


def test_canonicalize_and_bad_paths() -> None:
    assert canonicalize_project_path("Rec") == "lhcb/Rec"
    assert canonicalize_project_path("lhcb/reco") == "lhcb/Rec"
    assert canonicalize_project_path("lhcb/res") == "lhcb/Rec"
    assert canonicalize_project_path("moore") == "lhcb/Moore"
    assert canonicalize_project_path("lhcb/Moore") == "lhcb/Moore"

    assert find_bad_project_paths('{"project": "lhcb/reco"}') == ["lhcb/reco"]
    assert find_bad_project_paths('{"project": "lhcb/res"}') == ["lhcb/res"]
    assert find_bad_project_paths('{"project": "lhcb/Rec"}') == []
    assert find_bad_project_paths("searched lhcb/Moore and lhcb/Allen") == []
    assert find_bad_project_paths("lhcb/Rec Phys/VertexFit") == []


def test_annotate_user_question_does_not_inject_hints() -> None:
    q = "Search Rec and Moore for ParticleAdder and RecoConf ParticleCombiner"
    assert annotate_user_question(q) == q
    assert annotate_user_question("  spaced  ") == "spaced"
    assert annotate_user_question("What is 2+2?") == "What is 2+2?"


@pytest.mark.asyncio
async def test_session_ask_does_not_inject_hints() -> None:
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "test-key",
            "CERNGITLAB_CHAT_MODEL": "test-model",
            "CERNGITLAB_CHAT_MCP_COMMAND": str(Path("cerngitlab-mcp")),
        },
        clear=False,
    ):
        from cerngitlab_chat.config import ChatSettings
        from cerngitlab_chat.session import ChatSession

        settings = ChatSettings(_env_file=None)
        fake_agent = MagicMock()
        fake_agent.__aenter__ = AsyncMock(return_value=fake_agent)
        fake_agent.__aexit__ = AsyncMock(return_value=None)
        result = MagicMock()
        result.output = "ok"
        result.all_messages.return_value = []
        result.new_messages.return_value = []
        fake_agent.run = AsyncMock(return_value=result)

        with patch("cerngitlab_chat.session.create_agent", return_value=fake_agent):
            async with ChatSession.from_settings(settings) as session:
                await session.ask("Find ParticleAdder in Rec")

        prompt = fake_agent.run.await_args.args[0]
        assert prompt == "Find ParticleAdder in Rec"
        assert "Tool routing hint" not in prompt


def test_eval_questions_cover_many_topics() -> None:
    topics = {q["topic"] for q in EVAL_QUESTIONS}
    assert len(EVAL_QUESTIONS) >= 15
    for needed in (
        "reconstruction",
        "trigger",
        "analysis",
        "framework",
        "simulation",
        "geometry",
        "stack",
        "tooling",
    ):
        assert needed in topics, f"missing topic {needed}"
    ids = [q["id"] for q in EVAL_QUESTIONS]
    assert len(ids) == len(set(ids))


def test_score_good_particle_adder_chi2dof() -> None:
    spec = _spec("rec_particle_adder_chi2dof")
    reply = (
        "ParticleAdder in lhcb/Rec at Phys/VertexFit/src/ParticleAdder.cpp implements "
        "IParticleCombiner and adds daughter 4-momenta. It is not a geometric vertex "
        "fitter, so F.CHI2DOF vertex-quality cuts with ParticleAdder are inappropriate."
    )
    tools = [
        {
            "kind": "call",
            "tool": "search_code",
            "detail": '{"project": "lhcb/Rec", "search_term": "ParticleAdder"}',
        },
        {
            "kind": "call",
            "tool": "search_code",
            "detail": '{"project": "lhcb/Moore", "search_term": "ParticleCombiner"}',
        },
        {
            "kind": "call",
            "tool": "get_file_content",
            "detail": '{"project": "lhcb/Rec", "file_path": "Phys/VertexFit/src/ParticleAdder.cpp"}',
        },
    ]
    result = score_answer(question_spec=spec, reply=reply, tool_logs=tools)
    assert result.passed, result.issues
    assert result.score >= 85


def test_score_bad_wrong_project_and_hallucination() -> None:
    spec = _spec("rec_particle_adder_chi2dof")
    reply = (
        "ParticleAdder always runs CHI2DOF overfitting to spurious vertices in "
        "RecombineDecayTrees under lhcb/reco."
    )
    tools = [
        {
            "kind": "call",
            "tool": "search_code",
            "detail": '{"project": "lhcb/reco", "search_term": "ParticleAdder"}',
        },
    ]
    result = score_answer(question_spec=spec, reply=reply, tool_logs=tools)
    assert not result.passed
    assert result.score < 70
    assert any("FAIL" in i for i in result.issues)


def test_score_bad_missing_file_read() -> None:
    spec = _spec("rec_particle_adder_location")
    reply = (
        "ParticleAdder is somewhere in VertexFit and implements IParticleCombiner "
        "in lhcb/Rec."
    )
    tools = [
        {
            "kind": "call",
            "tool": "search_code",
            "detail": '{"project": "lhcb/Rec", "search_term": "ParticleAdder"}',
        },
    ]
    result = score_answer(question_spec=spec, reply=reply, tool_logs=tools)
    assert not result.passed
    assert any("get_file_content" in i for i in result.issues)


def test_score_good_davinci_dtf() -> None:
    spec = _spec("davinci_dtf_tutorial")
    reply = (
        "In lhcb/DaVinci tutorial6_DecayTreeFit.py, DecayTreeFitter is constructed with "
        "name, input_particles, mass_constraints, and optional input_pvs."
    )
    tools = [
        {
            "kind": "call",
            "tool": "search_code",
            "detail": '{"project": "lhcb/DaVinci", "search_term": "DecayTreeFitter"}',
        },
        {
            "kind": "call",
            "tool": "get_file_content",
            "detail": (
                '{"project": "lhcb/DaVinci", '
                '"file_path": "DaVinciExamples/python/DaVinciExamples/tupling/'
                'tutorial/tutorial6_DecayTreeFit.py"}'
            ),
        },
    ]
    result = score_answer(question_spec=spec, reply=reply, tool_logs=tools)
    assert result.passed, result.issues


def test_score_good_moore_readme() -> None:
    spec = _spec("moore_readme")
    reply = (
        "Moore (lhcb/Moore) is the LHCb high-level trigger application configuring "
        "HLT1/HLT2 reconstruction and selection lines."
    )
    tools = [
        {
            "kind": "call",
            "tool": "get_project_readme",
            "detail": '{"project": "lhcb/Moore"}',
        },
    ]
    result = score_answer(question_spec=spec, reply=reply, tool_logs=tools)
    assert result.passed, result.issues


def test_score_stack_list_without_tools() -> None:
    spec = _spec("stack_projects_list")
    reply = (
        "Default stack projects include lhcb/Moore, lhcb/Allen, lhcb/Gaudi, "
        "lhcb/Rec, lhcb/LHCb, lhcb/Gauss, and lhcb/DaVinci."
    )
    result = score_answer(question_spec=spec, reply=reply, tool_logs=[], used_tools=False)
    assert result.passed, result.issues


def test_score_recoconf_not_project() -> None:
    spec = _spec("recoconf_not_a_project")
    reply = (
        "RecoConf is a package under Rec/Moore, not a GitLab project. Search "
        "ParticleCombiner code in lhcb/Rec (and lhcb/Moore), not as lhcb/RecoConf."
    )
    result = score_answer(question_spec=spec, reply=reply, tool_logs=[], used_tools=False)
    assert result.passed, result.issues


def test_score_error_is_zero() -> None:
    spec = _spec("moore_hlt2conf")
    result = score_answer(
        question_spec=spec, reply="", tool_logs=[], error="timeout"
    )
    assert result.score == 0
    assert not result.passed


def test_score_good_allen_and_detector_topics() -> None:
    allen = _spec("allen_configuration")
    allen_reply = (
        "In lhcb/Allen, HLT1 sequences are configured under configuration/ "
        "areas such as configuration/sequences; cite a concrete sequence path "
        "from the search results."
    )
    allen_tools = [
        {
            "kind": "call",
            "tool": "search_code",
            "detail": '{"project": "lhcb/Allen", "search_term": "sequence"}',
        },
        {
            "kind": "call",
            "tool": "list_project_files",
            "detail": '{"project": "lhcb/Allen", "path": "configuration"}',
        },
    ]
    assert score_answer(
        question_spec=allen, reply=allen_reply, tool_logs=allen_tools
    ).passed

    det = _spec("detector_dd4hep")
    det_reply = (
        "lhcb/Detector contains DD4hep compact/geometry descriptions; for example "
        "a Detector/compact XML configures subdetector geometry."
    )
    det_tools = [
        {
            "kind": "call",
            "tool": "search_code",
            "detail": '{"project": "lhcb/Detector", "search_term": "DD4hep"}',
        },
    ]
    assert score_answer(
        question_spec=det, reply=det_reply, tool_logs=det_tools
    ).passed


def test_score_good_gauss_and_gaudi() -> None:
    gauss = _spec("gauss_decfiles")
    gauss_reply = (
        "lhcb/Gauss references decay files / event types in its configuration; "
        "cite the concrete path returned by search_code."
    )
    gauss_tools = [
        {
            "kind": "call",
            "tool": "search_code",
            "detail": '{"project": "lhcb/Gauss", "search_term": "DecFile"}',
        },
    ]
    assert score_answer(
        question_spec=gauss, reply=gauss_reply, tool_logs=gauss_tools
    ).passed

    gaudi = _spec("gaudi_algorithm")
    gaudi_reply = (
        "In lhcb/Gaudi, Algorithm is the base for processing algorithms; cite a "
        "header such as GaudiKernel/include/Gaudi/Algorithm.h from the project."
    )
    gaudi_tools = [
        {
            "kind": "call",
            "tool": "search_code",
            "detail": '{"project": "lhcb/Gaudi", "search_term": "class Algorithm"}',
        },
        {
            "kind": "call",
            "tool": "get_file_content",
            "detail": '{"project": "lhcb/Gaudi", "file_path": "GaudiKernel/include/Gaudi/Algorithm.h"}',
        },
    ]
    assert score_answer(
        question_spec=gaudi, reply=gaudi_reply, tool_logs=gaudi_tools
    ).passed


def test_hard_quota_error_not_retryable() -> None:
    from cerngitlab_chat.session import _is_retryable_error, is_hard_quota_error

    daily = Exception(
        "status_code: 429 ... free-models-per-day ... free_tier_daily"
    )
    soft = Exception("status_code: 429 rate limit too many requests")
    assert is_hard_quota_error(daily)
    assert not _is_retryable_error(daily)
    assert _is_retryable_error(soft)


def test_canonical_projects_match_stack_defaults() -> None:
    for name in ("moore", "allen", "gaudi", "rec", "davinci", "gauss", "detector"):
        assert name in CANONICAL_PROJECTS
        assert CANONICAL_PROJECTS[name].startswith("lhcb/")
