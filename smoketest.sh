#!/usr/bin/env bash
set -Eeuo pipefail

# Smoke test for lmsys-query-analysis
# Runs a small, end-to-end check using a temp SQLite DB and Chroma dir.
# Usage:
#   bash smoketest.sh                # full run (requires network + HF access)
#   SMOKE_LIMIT=500 bash smoketest.sh
#   SMOKE_OFFLINE=1 bash smoketest.sh  # seed minimal data if load fails

UV_CMD=${UV_CMD:-uv}
LIMIT=${SMOKE_LIMIT:-500}
N_CLUSTERS=${SMOKE_CLUSTERS:-8}
OFFLINE_FALLBACK=${SMOKE_OFFLINE:-1}

log()  { printf "[smoke] %s\n" "$*"; }
ok()   { printf "[smoke] ✅ %s\n" "$*"; }
warn() { printf "[smoke] ⚠️  %s\n" "$*"; }
die()  { printf "[smoke] ❌ %s\n" "$*"; exit 1; }

require_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"; }

require_cmd "$UV_CMD"

TMP_DIR=$(mktemp -d -t lmsys-smoke-XXXXXX)
DB_PATH="$TMP_DIR/queries.db"
CHROMA_PATH="$TMP_DIR/chroma"
export DB_PATH CHROMA_PATH
export PYTHONUNBUFFERED=1

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

log "Temp dir: $TMP_DIR"

log "Check CLI help"
$UV_CMD run lmsys --help >/dev/null
ok "CLI help works"

log "Attempt data load (limit=$LIMIT) with Chroma"
if $UV_CMD run lmsys load \
  --limit "$LIMIT" \
  --use-chroma \
  --db-path "$DB_PATH" \
  --chroma-path "$CHROMA_PATH"; then
  ok "Loaded dataset into SQLite + Chroma"
else
  if [ "$OFFLINE_FALLBACK" = "1" ]; then
    warn "Load failed; seeding minimal offline data into DB"
    $UV_CMD run python - <<'PY'
from sqlmodel import Session
from lmsys_query_analysis.db.connection import Database
from lmsys_query_analysis.db.models import Query
import os

db = Database(os.environ["DB_PATH"])  # pragma: no cover
db.create_tables()
texts = [
    "How do I install Python?",
    "What is a neural network?",
    "Explain gradient descent",
    "Hola, como aprendo Python?",
    "Best way to parse JSON in Go?",
    "What is SQLModel?",
    "Tips for clustering text data",
    "Troubleshooting CUDA install on Linux",
]
with db.get_session() as s:
    for i, t in enumerate(texts, 1):
        s.add(Query(conversation_id=f"smoke-{i}", model="gpt-4", query_text=t, language="en"))
    s.commit()
print("Seeded", len(texts), "queries")
PY
    ok "Seeded minimal data"
  else
    die "Load failed and offline fallback disabled"
  fi
fi

log "Run clustering (n_clusters=$N_CLUSTERS) with Chroma (KMeans)"
if $UV_CMD run lmsys cluster kmeans \
  --n-clusters "$N_CLUSTERS" \
  --use-chroma \
  --db-path "$DB_PATH" \
  --chroma-path "$CHROMA_PATH"; then
  ok "Clustering finished"
else
  die "Clustering failed (check model download / network)"
fi

log "Resolve latest run_id via Python"
RUN_ID=$($UV_CMD run python - <<'PY'
from sqlmodel import select
from lmsys_query_analysis.db.connection import Database
from lmsys_query_analysis.db.models import ClusteringRun
import os
db = Database(os.environ['DB_PATH'])
with db.get_session() as s:
    runs = s.exec(select(ClusteringRun).order_by(ClusteringRun.created_at.desc())).all()
    print(runs[0].run_id if runs else "")
PY
)
test -n "$RUN_ID" || die "No run_id found after clustering"
ok "Run ID: $RUN_ID"

log "List clusters"
$UV_CMD run lmsys list-clusters "$RUN_ID" --db-path "$DB_PATH" --limit 10 >/dev/null || die "list-clusters failed"
ok "Listed clusters"

log "Pick a sample cluster_id"
export RUN_ID
CLUSTER_ID=$($UV_CMD run python - <<'PY'
from sqlmodel import select
from lmsys_query_analysis.db.connection import Database
from lmsys_query_analysis.db.models import QueryCluster
import os
db = Database(os.environ['DB_PATH'])
run_id = os.environ['RUN_ID']
with db.get_session() as s:
    row = s.exec(select(QueryCluster.cluster_id).where(QueryCluster.run_id==run_id)).first()
    print(row if row is not None else "0")
PY
)

log "Inspect cluster $CLUSTER_ID"
$UV_CMD run lmsys inspect "$RUN_ID" "$CLUSTER_ID" --db-path "$DB_PATH" --show-queries 3 >/dev/null || die "inspect failed"
ok "Inspected cluster"

log "Export results to CSV + JSON"
$UV_CMD run lmsys export "$RUN_ID" --db-path "$DB_PATH" --format csv --output "$TMP_DIR/export.csv" >/dev/null || die "export csv failed"
$UV_CMD run lmsys export "$RUN_ID" --db-path "$DB_PATH" --format json --output "$TMP_DIR/export.json" >/dev/null || die "export json failed"
ok "Exported CSV and JSON"

log "Try semantic search if Chroma present"
if [ -d "$CHROMA_PATH" ]; then
  if $UV_CMD run lmsys search "python" --search-type clusters --run-id "$RUN_ID" --chroma-path "$CHROMA_PATH" >/dev/null; then
    ok "Cluster search works"
  else
    warn "Cluster search failed (likely no Chroma data in offline mode)"
  fi
else
  warn "Chroma path missing, skipping search"
fi

if [ "${SMOKE_HDBSCAN:-0}" = "1" ]; then
  log "Run HDBSCAN clustering (tiny params)"
  if $UV_CMD run lmsys cluster hdbscan \
    --db-path "$DB_PATH" \
    --chroma-path "$CHROMA_PATH" \
    --use-chroma \
    --embed-batch-size 16 \
    --chunk-size 1000 \
    --min-cluster-size 5 >/dev/null; then
    ok "HDBSCAN clustering finished"
  else
    die "HDBSCAN clustering failed"
  fi
fi

ok "Smoke test completed successfully"
