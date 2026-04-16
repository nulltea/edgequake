#!/usr/bin/env bash
# EdgeQuake database backup script
# Usage: ./scripts/backup-db.sh [backup_dir]
#
# Creates a timestamped pg_dump backup of the edgequake database.
# Includes all tables, AGE graph data, pgvector data, and KV storage.
#
# Requires: docker with edgequake-postgres container running

set -euo pipefail

BACKUP_DIR="${1:-$HOME/.edgequake/backups}"
CONTAINER="edgequake-postgres"
DB_USER="edgequake"
DB_NAME="edgequake"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/edgequake_${TIMESTAMP}.sql.gz"

# Ensure backup directory exists
mkdir -p "$BACKUP_DIR"

# Check container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "ERROR: Container '${CONTAINER}' is not running"
    exit 1
fi

echo "Starting backup of '${DB_NAME}' database..."
echo "  Container: ${CONTAINER}"
echo "  Output:    ${BACKUP_FILE}"

# Run pg_dump inside the container, compress output
docker exec "$CONTAINER" pg_dump -U "$DB_USER" -d "$DB_NAME" \
    --no-owner --no-privileges \
    | gzip > "$BACKUP_FILE"

SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
echo "Backup complete: ${BACKUP_FILE} (${SIZE})"

# Clean up old backups (keep last 10)
BACKUP_COUNT=$(ls -1 "${BACKUP_DIR}"/edgequake_*.sql.gz 2>/dev/null | wc -l)
if [ "$BACKUP_COUNT" -gt 10 ]; then
    echo "Cleaning old backups (keeping last 10)..."
    ls -1t "${BACKUP_DIR}"/edgequake_*.sql.gz | tail -n +11 | xargs rm -f
fi

echo "Done."
