"""Eval package for cerngitlab-chat (not part of default CI)."""

from cerngitlab_chat.eval.questions import EVAL_QUESTIONS
from cerngitlab_chat.eval.score import ScoreResult, score_answer

__all__ = ["EVAL_QUESTIONS", "ScoreResult", "score_answer"]
