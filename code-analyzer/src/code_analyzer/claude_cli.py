"""Thin wrapper around the `claude` CLI.

Runs a single headless invocation against a repo directory with read-only
tools (Read, Grep, Glob), asks for structured JSON output validated against
a given schema, and returns the parsed structured_output payload.

Why CLI subprocess, not the Python Agent SDK:
- Subscription OAuth (`CLAUDE_CODE_OAUTH_TOKEN`) is the most-supported path
  via the CLI. The SDK is officially API-key-only.
- We never set --bare (which ignores OAuth).
- We scrub `ANTHROPIC_API_KEY` from env before spawning, since API-key auth
  would take precedence over the subscription token.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ClaudeCliResult:
    structured_output: dict
    cost_usd: float | None
    duration_ms: int
    num_turns: int | None
    raw_result_text: str | None


class ClaudeCliError(RuntimeError):
    """Raised when the claude CLI returns a non-success or invalid JSON."""


def run_claude(
    *,
    prompt: str,
    cwd: Path,
    schema: dict,
    allowed_tools: list[str] = ("Read", "Grep", "Glob"),
    max_turns: int = 30,
    model: str | None = None,
    timeout_s: int = 300,
) -> ClaudeCliResult:
    """Spawn a single headless `claude -p` and return parsed structured output.

    `cwd` must be the repo root — Claude's Read/Grep/Glob operate there.
    """
    if not cwd.exists() or not cwd.is_dir():
        raise ClaudeCliError(f"cwd does not exist or is not a directory: {cwd}")

    env = _scrubbed_env()
    # Isolate per-invocation config dir ONLY when we have an explicit OAuth
    # token in env — that way concurrent runs don't step on each other's
    # plugin cache / session files. Without the token we must fall back to
    # disk auth at `~/.claude/.credentials.json`, which requires NOT moving
    # CLAUDE_CONFIG_DIR to an empty dir. (Local-dev convenience.)
    use_isolated_config = bool(env.get("CLAUDE_CODE_OAUTH_TOKEN"))
    config_dir_cm = (
        tempfile.TemporaryDirectory(prefix="claude-run-")
        if use_isolated_config
        else _NullContext()
    )
    with config_dir_cm as config_dir:
        if use_isolated_config:
            env["CLAUDE_CONFIG_DIR"] = config_dir

        cmd = [
            "claude",
            "-p",
            prompt,
            "--output-format",
            "json",
            "--allowed-tools",
            ",".join(allowed_tools),
            "--json-schema",
            json.dumps(schema),
            # Accept tool use without prompting. Sandbox is enforced by
            # --allowed-tools (Read/Grep/Glob are read-only in-workspace).
            "--permission-mode",
            "bypassPermissions",
            "--max-turns",
            str(max_turns),
            "--no-session-persistence",
        ]
        if model:
            cmd.extend(["--model", model])

        logger.info(
            "invoking claude cli: cwd=%s model=%s max_turns=%d timeout=%ds config_dir=%s",
            cwd,
            model or "default",
            max_turns,
            timeout_s,
            config_dir,
        )

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as e:
            raise ClaudeCliError(f"claude cli timed out after {timeout_s}s") from e

    if proc.returncode != 0:
        # The CLI returns non-zero for auth errors (401/403), quota errors
        # (429), and tool errors — but the actual reason lives in the JSON
        # payload on stdout, not stderr. Parse stdout first so the caller
        # sees the real message.
        structured_err = _extract_error_from_stdout(proc.stdout)
        if structured_err is not None:
            raise ClaudeCliError(
                f"claude cli exited {proc.returncode}: {structured_err}"
            )
        stderr_tail = (proc.stderr or "")[-2000:]
        raise ClaudeCliError(
            f"claude cli exited {proc.returncode}. stderr: {stderr_tail} stdout: {proc.stdout[:500]}"
        )

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise ClaudeCliError(
            f"claude cli stdout is not JSON: {proc.stdout[:500]!r}"
        ) from e

    if payload.get("is_error"):
        raise ClaudeCliError(
            f"claude cli reported is_error=true: {payload.get('result')!r}"
        )

    structured = payload.get("structured_output")
    if not isinstance(structured, dict):
        raise ClaudeCliError(
            f"claude cli did not return structured_output (got: {type(structured).__name__})"
        )

    return ClaudeCliResult(
        structured_output=structured,
        cost_usd=payload.get("total_cost_usd"),
        duration_ms=payload.get("duration_ms", 0),
        num_turns=payload.get("num_turns"),
        raw_result_text=payload.get("result"),
    )


def _extract_error_from_stdout(stdout: str) -> str | None:
    """Pull the human-readable error out of a failing CLI run.

    The CLI emits a JSON object like
    `{"is_error": true, "api_error_status": 401, "result": "…"}` on stdout
    even when returncode != 0. Surface that message; the CLI's stderr is
    usually empty for auth/quota errors.
    """
    try:
        payload = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict) or not payload.get("is_error"):
        return None
    api_status = payload.get("api_error_status")
    result = payload.get("result") or payload.get("error")
    if api_status and result:
        return f"api_error_status={api_status}: {result}"
    return str(result or payload)


def _scrubbed_env() -> dict[str, str]:
    """Copy of os.environ with auth precedence conflicts removed.

    The claude CLI env-var precedence puts ANTHROPIC_API_KEY *above*
    CLAUDE_CODE_OAUTH_TOKEN, so a stray API key would bill against API
    credits instead of the user's subscription. We strip anything that
    could override the intended OAuth token.
    """
    drop = {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_MODEL",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
    }
    env = {k: v for k, v in os.environ.items() if k not in drop}
    if "CLAUDE_CODE_OAUTH_TOKEN" not in env:
        logger.warning(
            "CLAUDE_CODE_OAUTH_TOKEN not set in env — claude CLI will fail to auth"
        )
    return env


def cli_version() -> str | None:
    """Best-effort `claude --version` for /health."""
    try:
        proc = subprocess.run(
            ["claude", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if proc.returncode == 0:
            return (proc.stdout or proc.stderr).strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None
    return None


# Suppress "unused variable" on uuid import kept for future per-run session-id use.
_ = uuid


class _NullContext:
    """No-op context manager used when we want to skip config-dir isolation."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *_: object) -> None:  # noqa: D401
        return None
