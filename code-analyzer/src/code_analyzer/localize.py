"""Compose the batch localization prompt and validate findings."""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import ValidationError

from .claude_cli import ClaudeCliError, run_claude
from .schema import (
    FINDINGS_SCHEMA,
    AlgorithmInput,
    Finding,
)

logger = logging.getLogger(__name__)


def localize_all(
    repo_dir: Path,
    algorithms: list[AlgorithmInput],
    *,
    model: str | None = None,
    timeout_s: int = 300,
) -> tuple[list[Finding], float | None, int, int | None]:
    """Run one Claude session that locates every algorithm in one go.

    Returns (findings, cost_usd, duration_ms, num_turns). Findings with
    a file that doesn't exist on disk are dropped — the agent sometimes
    hallucinates when a truly matching function isn't present.
    """
    prompt = _build_prompt(algorithms)
    result = run_claude(
        prompt=prompt,
        cwd=repo_dir,
        schema=FINDINGS_SCHEMA,
        allowed_tools=["Read", "Grep", "Glob"],
        max_turns=40,
        model=model,
        timeout_s=timeout_s,
    )

    raw_findings = result.structured_output.get("findings", [])
    if not isinstance(raw_findings, list):
        raise ClaudeCliError(
            f"structured_output.findings is not a list: {type(raw_findings).__name__}"
        )

    valid: list[Finding] = []
    for raw in raw_findings:
        try:
            f = Finding(**raw)
        except ValidationError as e:
            logger.warning("dropping malformed finding %r: %s", raw, e)
            continue
        # Path-sanity: agent occasionally returns paths outside the repo
        # or non-existent files. Strip those.
        candidate = (repo_dir / f.file).resolve()
        repo_root = repo_dir.resolve()
        if not str(candidate).startswith(str(repo_root) + "/"):
            logger.warning(
                "dropping finding outside repo: algorithm=%s file=%s",
                f.algorithm_id,
                f.file,
            )
            continue
        if not candidate.is_file():
            logger.warning(
                "dropping finding with missing file: algorithm=%s file=%s",
                f.algorithm_id,
                f.file,
            )
            continue
        if f.end_line < f.start_line:
            logger.warning(
                "dropping finding with inverted line range: algorithm=%s %d-%d",
                f.algorithm_id,
                f.start_line,
                f.end_line,
            )
            continue
        valid.append(f)

    known_ids = {a.id for a in algorithms}
    for f in valid:
        if f.algorithm_id not in known_ids:
            logger.warning(
                "finding references unknown algorithm_id=%s; keeping anyway",
                f.algorithm_id,
            )

    logger.info(
        "localization complete: %d input algos → %d valid findings (cost=%s turns=%s)",
        len(algorithms),
        len(valid),
        result.cost_usd,
        result.num_turns,
    )
    return valid, result.cost_usd, result.duration_ms, result.num_turns


def _build_prompt(algorithms: list[AlgorithmInput]) -> str:
    """Compose a batch prompt describing every algorithm we want located.

    Explicit about using Grep first (cheapest), Read on shortlist, Glob to
    narrow by language. Tells the agent to skip algorithms it can't find
    rather than hallucinate a match.
    """
    lines: list[str] = [
        "You are given extracted algorithm descriptions from an academic paper.",
        "For EACH algorithm, locate the FUNCTION, METHOD, or CLASS in the current",
        "working directory's codebase that implements it.",
        "",
        "Strategy:",
        "  1. Use Grep on each algorithm's name and characteristic identifiers.",
        "  2. Use Glob to narrow by likely file extensions if a language hint is given.",
        "  3. Use Read to confirm the candidate function's body matches the description.",
        "  4. Return ONE best match per algorithm. Pick the primary implementation,",
        "     not a callsite or a test. If no plausible match exists, SKIP the algorithm",
        "     (omit it from findings) — do not invent a match.",
        "  5. Line numbers are 1-indexed, inclusive on both ends.",
        "     Span the full function/method/class definition.",
        "  6. Confidence: 'high' if names match and body aligns with the description;",
        "     'medium' if names partially match or the body differs from the paper;",
        "     'low' if you inferred from context without direct name match.",
        "",
        "Output: JSON matching the provided schema. Paths are relative to the repo root.",
        "",
        "ALGORITHMS TO LOCATE:",
        "",
    ]
    for i, algo in enumerate(algorithms, 1):
        lines.append(f"--- Algorithm {i} ---")
        lines.append(f"algorithm_id: {algo.id}")
        lines.append(f"name: {algo.name}")
        if algo.description:
            desc = algo.description.strip()
            if len(desc) > 2_000:
                desc = desc[:2_000] + "…"
            lines.append(f"description: {desc}")
        if algo.pseudocode:
            pc = algo.pseudocode.strip()
            if len(pc) > 1_500:
                pc = pc[:1_500] + "…"
            lines.append(f"pseudocode:\n{pc}")
        if algo.languages_hint:
            lines.append(f"languages_hint: {', '.join(algo.languages_hint)}")
        lines.append("")
    return "\n".join(lines)
