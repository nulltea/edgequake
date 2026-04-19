"""Shallow clone with size cap + license detection.

Keeps the clone on disk (on the shared volume) so edgequake can extract
function snippets afterwards by (file, start, end).
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


SPDX_GUESS = re.compile(
    r"^\s*(?:#\s*)?(?:SPDX-License-Identifier:\s*)?"
    r"(MIT|Apache-2\.0|BSD-3-Clause|BSD-2-Clause|GPL-[23]\.0|LGPL-[23]\.0|MPL-2\.0|"
    r"ISC|Unlicense|CC0-1\.0)\b",
    re.IGNORECASE,
)


def repo_dir_for(workspace: Path, repo_url: str, repo_commit: str) -> Path:
    """Deterministic path per (url, commit) so repeated requests reuse the clone."""
    key = f"{repo_url}@{repo_commit}".encode()
    sha = hashlib.sha1(key).hexdigest()[:16]  # nosec: not a secret, just a filename
    return workspace / sha


def detect_license(root: Path) -> str | None:
    for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING", "COPYING.md"):
        f = root / name
        if not f.exists():
            continue
        try:
            text = f.read_text(errors="replace")[:4000]
        except OSError:
            continue
        m = SPDX_GUESS.search(text)
        if m:
            return m.group(1).upper().replace("BSD-3-CLAUSE", "BSD-3-Clause").replace(
                "BSD-2-CLAUSE", "BSD-2-Clause"
            )
        # heuristic: bare "MIT License" header
        if text.lower().startswith("mit license"):
            return "MIT"
        if "apache license" in text.lower()[:200]:
            return "Apache-2.0"
    return None


def folder_size_mb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total / (1024 * 1024)


def shallow_clone(
    repo_url: str,
    repo_commit: str,
    dest: Path,
    size_cap_mb: int,
    timeout_s: int,
) -> tuple[Path, str, str | None]:
    """Clone `repo_url@repo_commit` into `dest`. Idempotent: reuses an existing clone.

    Returns (path, resolved_commit_sha, license).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if not dest.exists():
        logger.info("clone start url=%s commit=%s dest=%s", repo_url, repo_commit, dest)
        cmd = [
            "git",
            "clone",
            "--depth=1",
            "--filter=blob:none",
            "--single-branch",
        ]
        if repo_commit and repo_commit not in ("HEAD", "main", "master"):
            # Can't --depth=1 with an arbitrary sha directly; need to fetch it separately.
            _init_and_fetch_sha(repo_url, repo_commit, dest, timeout_s)
        else:
            cmd.extend([repo_url, str(dest)])
            _run(cmd, timeout_s=timeout_s)

        actual_size = folder_size_mb(dest)
        if actual_size > size_cap_mb:
            logger.warning(
                "repo %s too large (%.1f MB > cap %d MB), removing clone",
                repo_url,
                actual_size,
                size_cap_mb,
            )
            shutil.rmtree(dest, ignore_errors=True)
            raise ValueError(
                f"Repo {repo_url} exceeds size cap "
                f"({actual_size:.1f} MB > {size_cap_mb} MB)"
            )
    else:
        logger.info("clone reuse dest=%s", dest)

    resolved_sha = _run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"], timeout_s=30
    ).stdout.strip()
    return dest, resolved_sha, detect_license(dest)


def _init_and_fetch_sha(
    repo_url: str, sha: str, dest: Path, timeout_s: int
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    _run(["git", "-C", str(dest), "init", "-q"], timeout_s=30)
    _run(
        ["git", "-C", str(dest), "remote", "add", "origin", repo_url],
        timeout_s=30,
    )
    _run(
        [
            "git",
            "-C",
            str(dest),
            "fetch",
            "--depth=1",
            "--filter=blob:none",
            "origin",
            sha,
        ],
        timeout_s=timeout_s,
    )
    _run(
        ["git", "-C", str(dest), "checkout", "-q", "FETCH_HEAD"], timeout_s=60
    )


def _run(cmd: list[str], *, timeout_s: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout_s,
    )
