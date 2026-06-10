#!/usr/bin/env bash
#
# Backfill parsed references for every document in the default workspace.
#
# Re-parses each document's stored markdown through the deterministic reference
# parser and writes rows into `document_references`. It does NOT reprocess
# documents (no chunking / embedding / LLM work) and is idempotent.
#
# Usage:
#   scripts/backfill_references.sh [--dry-run]
#
# Env (overrides):
#   DATABASE_URL      Postgres connection string. Falls back to a local .env
#                     (this dir's parent or repo root) if not already set.
#   EQ_WORKSPACE_ID   Workspace UUID (default: built-in default workspace).
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load DATABASE_URL from a .env if not already in the environment.
if [[ -z "${DATABASE_URL:-}" ]]; then
  for env_file in "${CRATE_DIR}/.env" "${CRATE_DIR}/../.env"; do
    if [[ -f "${env_file}" ]]; then
      # shellcheck disable=SC1090
      set -a; source "${env_file}"; set +a
      break
    fi
  done
fi

if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "error: DATABASE_URL is not set (export it or add it to a .env)" >&2
  exit 1
fi

# --dry-run → set EQ_DRY_RUN for the binary.
for arg in "$@"; do
  case "${arg}" in
    --dry-run) export EQ_DRY_RUN=1 ;;
    *) echo "unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

echo "Running reference backfill (dry_run=${EQ_DRY_RUN:-0})..."
cd "${CRATE_DIR}"
exec cargo run --release --example backfill_references --features postgres
