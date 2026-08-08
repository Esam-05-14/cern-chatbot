# Code chatbot (`cerngitlab-chat`)

Optional **MCP-backed** LHCb / CERN GitLab code Q&A CLI + local web UI (Team A Option 1),
with optional **FAISS RAG** (`retrieve_code`) for hybrid answering (Option 2).

## Install

```bash
uv sync --group chat
# RAG indexer + retrieve_code:
uv sync --group rag
```

Or with pip: `pip install -e ".[chat]"` / `pip install -e ".[rag]"`.

On Windows, stop Cursor’s MCP server before `uv sync` if it locks `cerngitlab-mcp.exe`.
Prefer the full venv to avoid the locked default `.venv` and reduce security-prompt noise:

```powershell
.\.venv-full\Scripts\Activate.ps1
uv sync --group chat --group rag --active
```

Pin the MCP child if needed: `CERNGITLAB_CHAT_MCP_COMMAND` →
`.venv-full\Scripts\cerngitlab-mcp.exe`.

## Configuration

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | yes | API key for an OpenAI-compatible endpoint |
| `OPENAI_BASE_URL` | no | OpenAI-compatible base URL (LiteLLM, OpenRouter, …) |
| `CERNGITLAB_CHAT_MODEL` | no | Chat model id — **prefer a tool-calling model** (default `gpt-4.1-mini`) |
| `CERNGITLAB_CHAT_EMBED_MODEL` | no | Embedding model for indexing (default `text-embedding-3-small`) |
| `CERNGITLAB_CHAT_MAX_TOOL_ROUNDS` | no | Max model rounds per question (default `5`) |
| `CERNGITLAB_CHAT_MCP_COMMAND` | no | MCP executable override (auto-detects venv Scripts) |
| `CERNGITLAB_CHAT_MCP_RETRIES` | no | Extra retries on timeout/rate-limit errors (default `1`) |
| `CERNGITLAB_CHAT_RAG` | no | `1`/`true` to enable `retrieve_code` when an index exists |
| `CERNGITLAB_CHAT_RAG_INDEX_DIR` | no | FAISS dir (default `.cache/cerngitlab-chat/faiss`) |
| `CERNGITLAB_TOKEN` | recommended | GitLab token for code search / private projects |

### Model tip

Pin a tool-capable model. **Never** use `openrouter/free` for demos or evals (rate
limits + unreliable tool calling). Recommended: `gpt-4.1-mini` (OpenAI) or
`openai/gpt-4.1-mini` (OpenRouter). See [eval.md](eval.md).

### LiteLLM / OpenRouter example

```bash
export OPENAI_API_KEY=sk-or-v1-...
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
export CERNGITLAB_CHAT_MODEL=gpt-4.1-mini   # or another tool-capable id
export CERNGITLAB_TOKEN=glpat-...
```

## Usage

```bash
cerngitlab-chat
cerngitlab-chat -q "Where is ParticleAdder used in Moore?"
cerngitlab-chat --ui   # http://127.0.0.1:7860/
```

REPL commands: `/quit`, `/exit`, `/clear`.

### RAG index

```bash
# Smoke: one project, capped files/chunks
cerngitlab-chat-index --project lhcb/DaVinci --clone --max-files 40 --max-chunks 200

# Full default stack set (Moore, Allen, Gaudi, Rec, LHCb, Gauss, DaVinci)
cerngitlab-chat-index --clone
```

Then:

```bash
export CERNGITLAB_CHAT_RAG=1
cerngitlab-chat --ui
```

The agent should call `retrieve_code` first, then MCP to verify. The UI tool log shows both.

### Eval harness

Multi-topic questions (Rec, Moore, DaVinci, Allen, Gaudi, Gauss, Detector, stack/tooling)
with heuristic scoring. The system prompt is evidence-only (no primed physics answers):
use tools, cite files, and say you do not know when under-determined.

```bash
python -m cerngitlab_chat.eval.run_eval --list
python -m cerngitlab_chat.eval.run_eval --limit 5 --out eval_results.jsonl
python -m cerngitlab_chat.eval.run_eval --topic analysis --out eval_analysis.jsonl
```

MCP vs MCP+RAG comparison and scoring details: [eval.md](eval.md).

## Architecture

```text
User (CLI or Web UI) → cerngitlab-chat Agent → OpenAI-compatible LLM
                         ↓                ↓
              retrieve_code (FAISS)   MCPToolset → gitlab.cern.ch
```

Extension point: [`cerngitlab_chat.backends`](../src/cerngitlab_chat/backends.py)

- `McpStdioBackend` — live GitLab MCP
- `RagToolBackend` — local FAISS `retrieve_code` when `CERNGITLAB_CHAT_RAG=1`
- `CompositeToolBackend` — combines both
