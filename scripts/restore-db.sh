#!/usr/bin/env bash
# EdgeQuake database restore script
# Usage: ./scripts/restore-db.sh [backup_file]
#
# Restores a gzipped pg_dump backup into the edgequake database.
# If no backup file is given, the most recent file in ~/.edgequake/backups is used.
#
# DESTRUCTIVE: drops and recreates the 'edgequake' database.
# Requires: docker with edgequake-postgres container running.

set -euo pipefail

BACKUP_DIR="${EDGEQUAKE_BACKUP_DIR:-$HOME/.edgequake/backups}"
CONTAINER="edgequake-postgres"
API_CONTAINER="edgequake"
DB_USER="edgequake"
DB_NAME="edgequake"

BACKUP_FILE="${1:-}"
if [[ -z "${BACKUP_FILE}" ]]; then
    BACKUP_FILE=$(ls -t "${BACKUP_DIR}"/edgequake_*.sql.gz 2>/dev/null | head -n1 || true)
    if [[ -z "${BACKUP_FILE}" ]]; then
        echo "ERROR: No backups found in ${BACKUP_DIR}" >&2
        exit 1
    fi
fi

if [[ ! -f "${BACKUP_FILE}" ]]; then
    echo "ERROR: Backup file not found: ${BACKUP_FILE}" >&2
    exit 1
fi

if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "ERROR: Container '${CONTAINER}' is not running" >&2
    exit 1
fi

SIZE=$(du -h "${BACKUP_FILE}" | cut -f1)
echo "About to restore:"
echo "  Backup:    ${BACKUP_FILE} (${SIZE})"
echo "  Container: ${CONTAINER}"
echo "  Database:  ${DB_NAME} (will be DROPPED and recreated)"
echo ""
read -r -p "Proceed? [y/N] " ans
if [[ ! "${ans}" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Stop API so it doesn't hold connections or try to migrate mid-restore.
if docker ps --format '{{.Names}}' | grep -q "^${API_CONTAINER}$"; then
    echo "Stopping ${API_CONTAINER}..."
    docker stop "${API_CONTAINER}" >/dev/null
    RESTART_API=1
else
    RESTART_API=0
fi

echo "Dropping and recreating database..."
docker exec "${CONTAINER}" psql -U "${DB_USER}" -d postgres -c \
    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='${DB_NAME}' AND pid<>pg_backend_pid();" \
    >/dev/null
docker exec "${CONTAINER}" psql -U "${DB_USER}" -d postgres -c "DROP DATABASE IF EXISTS ${DB_NAME};"
docker exec "${CONTAINER}" psql -U "${DB_USER}" -d postgres -c "CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};"

echo "Restoring data..."
gunzip -c "${BACKUP_FILE}" | docker exec -i "${CONTAINER}" psql -U "${DB_USER}" -d "${DB_NAME}" >/dev/null

# Sync Apache AGE graphids to restored namespace oids.
#
# pg_dump preserves the ag_catalog.ag_graph.graphid column, but graphid
# must equal pg_namespace.oid of the graph's schema. After restore, the
# schema is recreated with a fresh oid and AGE's cypher() function fails
# with "graph with oid N does not exist" on every query.
#
# Fix: for each graph whose graphid no longer matches its namespace oid,
# drop the ag_label FK, UPDATE both tables, re-add the FK.
echo "Syncing AGE graph oids..."
docker exec -i "${CONTAINER}" psql -U "${DB_USER}" -d "${DB_NAME}" -v ON_ERROR_STOP=1 >/dev/null <<'SQL'
DO $$
DECLARE
    has_mismatch boolean;
    g RECORD;
BEGIN
    -- Skip silently if AGE isn't installed.
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'age') THEN
        RETURN;
    END IF;

    SELECT EXISTS (
        SELECT 1 FROM ag_catalog.ag_graph WHERE graphid <> namespace::oid
    ) INTO has_mismatch;

    IF NOT has_mismatch THEN
        RETURN;
    END IF;

    ALTER TABLE ag_catalog.ag_label DROP CONSTRAINT fk_graph_oid;
    FOR g IN
        SELECT graphid AS old_id, namespace::oid AS new_id
        FROM ag_catalog.ag_graph
        WHERE graphid <> namespace::oid
    LOOP
        UPDATE ag_catalog.ag_label SET graph = g.new_id WHERE graph = g.old_id;
        UPDATE ag_catalog.ag_graph SET graphid = g.new_id WHERE graphid = g.old_id;
    END LOOP;
    ALTER TABLE ag_catalog.ag_label
        ADD CONSTRAINT fk_graph_oid
        FOREIGN KEY (graph) REFERENCES ag_catalog.ag_graph(graphid);
END $$;
SQL

if [[ "${RESTART_API}" == "1" ]]; then
    echo "Restarting ${API_CONTAINER}..."
    docker start "${API_CONTAINER}" >/dev/null
fi

echo "Restore complete."
