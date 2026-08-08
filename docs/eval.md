# Chatbot RAG / hybrid eval notes

## Model guidance (tool calling)

**Required for publish / demos:** pin a concrete **tool-capable** chat model.

Do **not** use `openrouter/free` (or other bare free routers): they randomly pick
upstreams, skip tools, and hit daily free-tier 429s.

```env
# OpenAI
CERNGITLAB_CHAT_MODEL=gpt-4.1-mini

# OpenRouter — pin a specific tool-capable model id (not openrouter/free)
CERNGITLAB_CHAT_MODEL=openai/gpt-4.1-mini
OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

`cerngitlab-chat` prints a stderr warning when a weak free/router model id is configured.

Embeddings (indexer) need an embeddings-capable model id:

```env
CERNGITLAB_CHAT_EMBED_MODEL=text-embedding-3-small
```

## MCP vs MCP+RAG comparison

1. Build an index (smoke with one project):

```bash
uv sync --group rag
cerngitlab-chat-index --project lhcb/DaVinci --clone --max-files 40 --max-chunks 200
```

2. Run eval **MCP-only**:

```bash
# CERNGITLAB_CHAT_RAG unset / 0
python -m cerngitlab_chat.eval.run_eval --limit 5 --out eval_mcp.jsonl
```

3. Run eval **MCP+RAG**:

```bash
set CERNGITLAB_CHAT_RAG=1
python -m cerngitlab_chat.eval.run_eval --limit 5 --out eval_mcp_rag.jsonl
```

4. Compare JSONL rows: `used_tools`, whether `retrieve_code` appears in `tool_logs`,
latency, heuristic `score` / `passed`, and citation quality vs `reference`.

## Heuristic scoring

Live runs attach a 0–100 score (pass ≥ 70, no hard `FAIL:` issues). Checks include:

- tools used (except a few meta questions)
- no bad project paths (`lhcb/reco`, `lhcb/res`, wrong casing)
- `get_file_content` when the question requires it
- keyword / project citations from the question bank
- forbidden hallucination phrases (e.g. invented ParticleAdder/CHI2DOF stories)

```bash
python -m cerngitlab_chat.eval.run_eval --list
python -m cerngitlab_chat.eval.run_eval --topic reconstruction --limit 2 --out eval_rec.jsonl
python -m cerngitlab_chat.eval.run_eval --score-only eval_rec.jsonl

# After a rate-limit abort: keep PASSes and retry the rest
python -m cerngitlab_chat.eval.run_eval --resume eval_live_full.jsonl --out eval_live_full.jsonl

# Or only specific ids
python -m cerngitlab_chat.eval.run_eval --ids davinci_dtf_tutorial,allen_configuration --out eval_subset.jsonl
```

If OpenRouter returns `free-models-per-day` / 429 or `requires more credits` / 402,
the harness aborts remaining questions (no pointless retries). Switch
`CERNGITLAB_CHAT_MODEL` to a tool-capable model with available credits (or wait for
the daily free reset), then `--resume`.

Unit coverage for prompts, path routing, and good/bad transcripts: `tests/test_chat_quality.py`.

The web UI tool log already shows `retrieve_code` calls/results when RAG is enabled.
