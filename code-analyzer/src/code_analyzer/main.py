"""FastAPI entrypoint for the code-analyzer service.

POST /analyze {repo_url, repo_commit?, algorithms[]}
  → clones repo into the shared workspace volume
  → one batched `claude` CLI invocation
  → returns findings ready for edgequake to chunk + embed.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException

from .claude_cli import ClaudeCliError, cli_version
from .clone import repo_dir_for, shallow_clone
from .localize import localize_all
from .schema import (
    AnalyzeRequest,
    AnalyzeResponse,
    HealthResponse,
)

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

WORKSPACE = Path(os.environ.get("CODE_ANALYZER_WORKSPACE", "/workspace"))

app = FastAPI(title="code-analyzer", version="0.1.0")


@app.on_event("startup")
async def _startup() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    logger.info("code-analyzer starting; workspace=%s", WORKSPACE)
    if not os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        logger.warning(
            "CLAUDE_CODE_OAUTH_TOKEN is not set — /analyze calls will fail. "
            "Run `claude setup-token` on the host and set this env var."
        )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        claude_cli_version=cli_version(),
        workspace=str(WORKSPACE),
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    if not req.algorithms:
        raise HTTPException(
            status_code=400, detail="at least one algorithm required"
        )
    start = time.monotonic()
    dest = repo_dir_for(WORKSPACE, req.repo_url, req.repo_commit)

    try:
        repo_path, resolved_sha, license_ = shallow_clone(
            req.repo_url,
            req.repo_commit,
            dest,
            size_cap_mb=req.size_cap_mb,
            timeout_s=req.timeout_s,
        )
    except ValueError as e:
        # Size-cap exceeded.
        raise HTTPException(status_code=413, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        logger.exception("clone failed")
        raise HTTPException(
            status_code=502, detail=f"clone failed: {e}"
        ) from e

    try:
        findings, cost, _, num_turns = localize_all(
            repo_path,
            req.algorithms,
            model=req.model,
            timeout_s=req.timeout_s,
        )
    except ClaudeCliError as e:
        logger.warning("claude cli error: %s", e)
        raise HTTPException(
            status_code=502, detail=f"claude cli error: {e}"
        ) from e
    except Exception as e:  # noqa: BLE001
        logger.exception("localization failed")
        raise HTTPException(
            status_code=500, detail=f"localization failed: {e}"
        ) from e

    duration_ms = int((time.monotonic() - start) * 1000)
    return AnalyzeResponse(
        repo_path=str(repo_path),
        repo_commit=resolved_sha,
        repo_license=license_,
        findings=findings,
        usage_cost_usd_equivalent=cost,
        duration_ms=duration_ms,
        num_turns=num_turns,
    )
