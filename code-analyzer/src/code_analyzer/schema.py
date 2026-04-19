"""Pydantic request/response models + the JSON schema we hand to Claude.

The schema Claude returns is the *inner* array of findings. Pydantic models
wrap that with service-level metadata (cost, timings, repo state).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ── Request ────────────────────────────────────────────────────────────────


class AlgorithmInput(BaseModel):
    id: str
    name: str
    description: str = ""
    pseudocode: str | None = None
    languages_hint: list[str] = Field(default_factory=list)


class AnalyzeRequest(BaseModel):
    repo_url: str
    repo_commit: str = "HEAD"
    algorithms: list[AlgorithmInput]
    size_cap_mb: int = 500
    timeout_s: int = 300
    # Optional model override. "sonnet" / "opus" / "haiku" / full id.
    # Defaults to whatever the user's subscription prefers (claude CLI default).
    model: str | None = None


class SnapshotRequest(BaseModel):
    repo_url: str
    repo_commit: str = "HEAD"
    size_cap_mb: int = 500
    timeout_s: int = 300


# ── Response ───────────────────────────────────────────────────────────────


class Finding(BaseModel):
    algorithm_id: str
    file: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    rationale: str
    confidence: Literal["high", "medium", "low"] = "medium"


class AnalyzeResponse(BaseModel):
    repo_path: str
    repo_commit: str
    repo_license: str | None = None
    findings: list[Finding]
    # Theoretical dollar cost if these tokens had been billed against
    # pay-as-you-go API credits. On a Claude Code subscription this is a
    # usage gauge, NOT an amount the user is charged. Use it as a signal
    # for how much subscription quota the call consumed.
    usage_cost_usd_equivalent: float | None = None
    duration_ms: int
    num_turns: int | None = None


class SnapshotResponse(BaseModel):
    repo_path: str
    repo_commit: str
    repo_license: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    claude_cli_version: str | None = None
    workspace: str


# ── JSON schema handed to Claude ───────────────────────────────────────────
#
# The CLI's --json-schema validates the top-level response. We ask Claude to
# return {"findings": [Finding, ...]} so multi-algorithm batches fit in one
# object.

FINDINGS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "algorithm_id": {"type": "string"},
                    "file": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                    "rationale": {"type": "string"},
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                },
                "required": [
                    "algorithm_id",
                    "file",
                    "start_line",
                    "end_line",
                    "rationale",
                    "confidence",
                ],
            },
        }
    },
    "required": ["findings"],
}
