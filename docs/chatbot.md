# Code chatbot (`cerngitlab-chat`)

Optional **MCP-only** LHCb / CERN GitLab code Q&A CLI (Team A Option 1).  
It drives this package’s MCP tools from a pydantic-ai agent over an OpenAI-compatible API. RAG is not included yet; the agent is shaped so a future `retrieve_code` tool can plug in (Option 2).

## Install

From the repo root:

```bash
uv sync --group chat
```

Or with pip:

```bash
pip install -e ".[chat]"
```

## Configuration

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | yes | API key for OpenAI, LiteLLM, Groq, vLLM, etc. |
| `OPENAI_BASE_URL` | no | OpenAI-compatible base URL (e.g. LiteLLM gateway) |
| `CERNGITLAB_CHAT_MODEL` | no | Model id (default `gpt-4.1-mini`) |
| `CERNGITLAB_CHAT_MAX_TOOL_ROUNDS` | no | Max model rounds per question (default `5`) |
| `CERNGITLAB_CHAT_MCP_COMMAND` | no | Override MCP executable (default: `cerngitlab-mcp-stdio` on PATH) |
| `CERNGITLAB_TOKEN` | recommended | GitLab token for code search / private projects |
| Other `CERNGITLAB_*` | no | Passed through to the MCP subprocess |

Values can live in a project `.env` file (same pattern as the MCP server).

### LiteLLM example

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://your-litellm-host/v1
export CERNGITLAB_CHAT_MODEL=gpt-4.1-mini
export CERNGITLAB_TOKEN=glpat-...
```

## Usage

Interactive REPL:

```bash
cerngitlab-chat
```

One-shot:

```bash
cerngitlab-chat -q "Where is ParticleAdder used in Moore?"
```

Commands inside the REPL: `/quit` or `/exit`.

## Smoke test

1. Sync the `chat` dependency group and ensure `cerngitlab-mcp-stdio` is on PATH.
2. Set `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL` + `CERNGITLAB_TOKEN`).
3. Run:

   ```bash
   cerngitlab-chat -q "What is the LHCb software stack project for HLT2?"
   ```

4. Confirm the answer cites GitLab evidence (project/path). On failure, check MCP timeouts and `CERNGITLAB_RATE_LIMIT_PER_MINUTE`.

## Architecture

```text
User CLI → cerngitlab-chat Agent → OpenAI-compatible LLM
                 ↓
         MCPToolset(StdioTransport) → cerngitlab-mcp → gitlab.cern.ch
```

Extension point: [`cerngitlab_chat.backends`](../src/cerngitlab_chat/backends.py)

- `ToolBackend` — protocol for anything that returns pydantic-ai toolsets
- `McpStdioBackend` — Phase 1 (live GitLab MCP via `MCPToolset` + `StdioTransport`)
- `CompositeToolBackend.add(...)` — register extra backends before `create_agent`

## Roadmap: RAG (Option 2)

Not implemented yet. Intended steps:

1. Indexer CLI: LHCb stack projects → AST-aware chunks → embeddings → local FAISS/Chroma.
2. New backend that exposes `retrieve_code(query, top_k)` via a FunctionToolset.
3. `default_tool_backend()` (or CLI) registers MCP **and** RAG; prompt already mentions using RAG for speed and MCP for fresh file-accurate lookups.
4. Optional eval harness (Q&A + LLM judge) later.
