"""Run the sample Q&A eval harness against a live or dry-run chatbot session.

Examples::

    # Print the question set only
    python -m cerngitlab_chat.eval.run_eval --list

    # Live MCP-backed answers with heuristic scores; writes JSONL results
    python -m cerngitlab_chat.eval.run_eval --out eval_results.jsonl

    # Limit / filter / resume after rate limits
    python -m cerngitlab_chat.eval.run_eval --limit 3 --out eval_results.jsonl
    python -m cerngitlab_chat.eval.run_eval --ids davinci_dtf_tutorial,allen_configuration
    python -m cerngitlab_chat.eval.run_eval --resume eval_live_full.jsonl --out eval_live_full.jsonl

    # Re-score an existing JSONL (no live LLM)
    python -m cerngitlab_chat.eval.run_eval --score-only eval_results.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from cerngitlab_chat.config import get_chat_settings
from cerngitlab_chat.eval.questions import EVAL_QUESTIONS
from cerngitlab_chat.eval.score import score_answer
from cerngitlab_chat.session import ChatSession, is_hard_quota_error

_BY_ID = {str(q["id"]): q for q in EVAL_QUESTIONS}


def _attach_score(record: dict[str, object], spec: dict | None) -> None:
    if spec is None:
        record["score"] = None
        record["passed"] = False
        record["score_issues"] = ["unknown question id"]
        return
    result = score_answer(
        question_spec=spec,
        reply=str(record.get("reply") or ""),
        tool_logs=list(record.get("tool_logs") or []),  # type: ignore[arg-type]
        used_tools=bool(record.get("used_tools")),
        error=record.get("error") if isinstance(record.get("error"), str) else None,
    )
    record["score"] = result.score
    record["passed"] = result.passed
    record["score_issues"] = result.issues
    record["score_checks"] = result.checks


def _print_summary(records: list[dict[str, object]]) -> None:
    n = len(records)
    if not n:
        print("No records.")
        return
    passed = sum(1 for r in records if r.get("passed"))
    scores = [float(r["score"]) for r in records if isinstance(r.get("score"), (int, float))]
    mean = sum(scores) / len(scores) if scores else 0.0
    print(f"Summary: {passed}/{n} passed; mean score={mean:.1f}")
    for r in records:
        status = "PASS" if r.get("passed") else "FAIL"
        sid = r.get("id")
        sc = r.get("score")
        issues = r.get("score_issues") or []
        # Keep summary readable when provider dumps a huge 429 body.
        short_issues: list[str] = []
        for issue in issues:
            text = str(issue)
            if len(text) > 160:
                text = text[:157] + "..."
            short_issues.append(text)
        extra = f" issues={short_issues}" if short_issues and status == "FAIL" else ""
        print(f"  [{status}] {sid}: score={sc}{extra}")


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _select_questions(
    *,
    limit: int | None,
    topic: str | None,
    ids: list[str] | None,
    skip_ids: set[str],
) -> list[dict]:
    questions = list(EVAL_QUESTIONS)
    if topic:
        questions = [q for q in questions if q.get("topic") == topic]
    if ids:
        wanted = set(ids)
        questions = [q for q in questions if q["id"] in wanted]
        missing = wanted - {q["id"] for q in questions}
        if missing:
            raise SystemExit(f"Unknown question id(s): {sorted(missing)}")
    if skip_ids:
        questions = [q for q in questions if q["id"] not in skip_ids]
    if limit is not None:
        questions = questions[:limit]
    return questions


async def _run_live(
    *,
    limit: int | None,
    out: Path,
    topic: str | None,
    ids: list[str] | None,
    resume: Path | None,
) -> None:
    settings = get_chat_settings()
    prior = _load_jsonl(resume) if resume is not None else []
    # On resume: keep prior PASSes; re-run everything else (failed / missing).
    keep = [r for r in prior if r.get("passed")] if resume is not None else []
    skip_ids = {str(r["id"]) for r in keep}
    questions = _select_questions(
        limit=limit, topic=topic, ids=ids, skip_ids=skip_ids
    )
    if not questions:
        print("Nothing to run (all selected questions already passed).")
        _print_summary(keep or prior)
        return

    print(
        f"Running {len(questions)} question(s); "
        f"model={settings.model}; rag={settings.rag_enabled}; "
        f"kept_passes={len(keep)}"
    )
    # Start from full prior JSONL so --limit resume does not drop unevaluated rows.
    records_by_id = {str(r["id"]): r for r in prior}
    for r in keep:
        records_by_id[str(r["id"])] = r
    abort_remaining = False
    abort_reason = ""

    async with ChatSession.from_settings(settings) as session:
        for item in questions:
            if abort_remaining:
                record: dict[str, object] = {
                    "id": item["id"],
                    "topic": item.get("topic"),
                    "question": item["question"],
                    "reference": item["reference"],
                    "model": settings.model,
                    "rag_enabled": settings.rag_enabled,
                    "reply": "",
                    "used_tools": False,
                    "tool_logs": [],
                    "error": abort_reason,
                    "latency_s": 0.0,
                }
                _attach_score(record, item)
                records_by_id[str(item["id"])] = record
                print(f"{item['id']}: SKIP score=0.0 (quota abort)")
                continue

            started = time.perf_counter()
            record = {
                "id": item["id"],
                "topic": item.get("topic"),
                "question": item["question"],
                "reference": item["reference"],
                "model": settings.model,
                "rag_enabled": settings.rag_enabled,
            }
            try:
                turn = await session.ask(item["question"])
                record["reply"] = turn.reply
                record["used_tools"] = turn.used_tools
                record["tool_logs"] = [
                    {
                        "kind": e.kind,
                        "tool": e.tool,
                        "detail": e.detail[:400],
                    }
                    for e in turn.tool_logs
                ]
                record["error"] = None
            except Exception as exc:  # noqa: BLE001
                record["reply"] = ""
                record["used_tools"] = False
                record["tool_logs"] = []
                record["error"] = str(exc)
                if is_hard_quota_error(exc):
                    abort_remaining = True
                    abort_reason = (
                        "aborted: provider quota exhausted "
                        f"({type(exc).__name__}: {str(exc)[:120]})"
                    )
                    print(
                        "Provider quota exhausted — skipping remaining questions. "
                        "Re-run with --resume after reset or switch CERNGITLAB_CHAT_MODEL."
                    )
            record["latency_s"] = round(time.perf_counter() - started, 3)
            _attach_score(record, item)
            records_by_id[str(item["id"])] = record
            status = "PASS" if record.get("passed") else "FAIL"
            print(
                f"{item['id']}: {status} score={record['score']} "
                f"tools={record.get('used_tools')} latency={record['latency_s']}s"
            )

    # Preserve original question order for output.
    ordered_ids = [str(q["id"]) for q in EVAL_QUESTIONS]
    if ids:
        ordered_ids = [i for i in ordered_ids if i in set(ids)]
    elif topic:
        ordered_ids = [
            str(q["id"]) for q in EVAL_QUESTIONS if q.get("topic") == topic
        ]
    # Include any kept passes + newly run ids
    final: list[dict[str, object]] = []
    seen: set[str] = set()
    for qid in ordered_ids:
        if qid in records_by_id:
            final.append(records_by_id[qid])
            seen.add(qid)
    for qid, rec in records_by_id.items():
        if qid not in seen:
            final.append(rec)

    _write_jsonl(out, final)
    print(f"Wrote {out}")
    _print_summary(final)


def _score_only(path: Path) -> None:
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            spec = _BY_ID.get(str(record.get("id")))
            _attach_score(record, spec)
            records.append(record)
    out = path.with_suffix(".scored.jsonl")
    with out.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {out}")
    _print_summary(records)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="cerngitlab-chat eval harness")
    parser.add_argument("--list", action="store_true", help="List questions and exit")
    parser.add_argument("--limit", type=int, default=None, help="Only first N questions")
    parser.add_argument("--topic", type=str, default=None, help="Filter by topic")
    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Comma-separated question ids to run",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Existing JSONL: keep PASSes and re-run the rest",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("eval_results.jsonl"),
        help="JSONL output path for live runs",
    )
    parser.add_argument(
        "--score-only",
        type=Path,
        default=None,
        help="Re-score an existing JSONL without calling the LLM",
    )
    args = parser.parse_args(argv)

    if args.list:
        for item in EVAL_QUESTIONS:
            print(f"{item['id']} [{item.get('topic')}]: {item['question']}")
        return

    if args.score_only is not None:
        _score_only(args.score_only)
        return

    ids = [s.strip() for s in args.ids.split(",") if s.strip()] if args.ids else None
    asyncio.run(
        _run_live(
            limit=args.limit,
            out=args.out,
            topic=args.topic,
            ids=ids,
            resume=args.resume,
        )
    )


if __name__ == "__main__":
    main()
