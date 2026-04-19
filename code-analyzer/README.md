# code-analyzer

FastAPI service that locates paper algorithms in a cloned repository by
driving the `claude` CLI in headless mode. Produces `(file, start_line,
end_line, rationale)` tuples for every algorithm the user supplies.

Used as a sidecar to edgequake in Phase 1 of the Reference Code GraphRAG
extension. The clone persists on a shared volume so edgequake can read the
located snippets directly afterwards.

## Auth — subscription-based

The service is designed to run under a **Claude Code subscription**, not
a pay-as-you-go Anthropic API key.

```bash
# One-time, on a machine with a browser:
claude setup-token
# → prints a 1-year sk-ant-oat01-... token. Copy it.
```

Put that into the docker-compose `.env`:

```
CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-...
```

**Do not** also set `ANTHROPIC_API_KEY` — the CLI prefers the API key and
would bill that account instead of your subscription. The service's env
scrubber removes it defensively, but keeping it out of the compose env is
the clean solution.

## Quota

Calls share your Claude Code subscription's 5-hour rolling window and
weekly cap with interactive use. Plan accordingly — a single paper
(5–10 algorithms) costs one Agent session, typically ~$0.10–$0.40 worth
of token usage.

## API

```
POST /analyze
  {
    "repo_url":     "https://github.com/owner/repo",
    "repo_commit":  "HEAD",
    "algorithms": [
       { "id": "<uuid>", "name": "...", "description": "...",
         "pseudocode": "...", "languages_hint": ["python"] }
    ],
    "size_cap_mb": 500,
    "timeout_s":   300,
    "model":       "sonnet"          // optional
  }
→ 200 {
    "repo_path":   "/workspace/<sha>",
    "repo_commit": "<resolved_sha>",
    "repo_license": "MIT",
    "findings": [
       { "algorithm_id": "...", "file": "model.py",
         "start_line": 29, "end_line": 76,
         "rationale": "...", "confidence": "high" }
    ],
    "cost_usd": 0.21,
    "duration_ms": 12_900,
    "num_turns": 4
  }

GET /health
  → { "status": "ok", "claude_cli_version": "2.1.114 (Claude Code)", "workspace": "/workspace" }
```

## Local dev (without docker)

```bash
cd code-analyzer
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
export CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-...
export CODE_ANALYZER_WORKSPACE=/tmp/code-analyzer-workspace
uvicorn code_analyzer.main:app --reload --port 9100
```

## Safety

- The Claude session runs with `--allowed-tools Read,Grep,Glob` — no
  writes, no shell execution, no network (beyond the LLM itself). Safe
  for untrusted code clones.
- Repositories are cloned `--depth=1 --filter=blob:none --single-branch`,
  capped at `size_cap_mb` (default 500 MB).
- Each analyze call gets a fresh `CLAUDE_CONFIG_DIR` so concurrent calls
  don't corrupt each other's plugin cache.

## Token rotation

`CLAUDE_CODE_OAUTH_TOKEN` is valid for one year. On 401 from the CLI, the
`/analyze` endpoint returns 502 with an error message pointing here.
Re-run `claude setup-token` on the host, update the `.env`, and
`docker compose up -d code-analyzer` to restart the container.
