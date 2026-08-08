## Summary

- **0.3.0 alpha**: optional `cerngitlab-chat` (MCP-backed CLI + local web UI), FAISS RAG MVP (`cerngitlab-chat-index` / `retrieve_code`), evidence-first prompts (no primed ParticleAdder/CHI2DOF answers).
- Eval harness with multi-topic questions, heuristic scoring, `--resume` / `--ids`, fail-fast on OpenRouter free-tier 429 and credit 402.
- Docs require a **pinned tool-capable model** (never `openrouter/free`); stderr warning when a weak router id is configured.
- Windows guidance: prefer `.venv-full` for MCP child resolution.

## Test plan

- [ ] `uv sync --group chat --group rag` (use `.venv-full` on Windows if `.venv` is locked)
- [ ] `pytest tests/test_chat.py tests/test_chat_quality.py tests/test_rag.py -k "not ui_static"`
- [ ] `cerngitlab-chat --help` / `cerngitlab-chat --ui` smoke with `CERNGITLAB_TOKEN` + pinned `CERNGITLAB_CHAT_MODEL`
- [ ] Optional: `python -m cerngitlab_chat.eval.run_eval --resume eval_live_full.jsonl` once provider credits are available

## Notes

Alpha/MVP — not a production guarantee for expert LHCb physics Q&A. Answers must be treated as tool-cited assistance.
