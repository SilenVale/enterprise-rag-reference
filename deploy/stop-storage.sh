#!/usr/bin/env bash
# 停掉本脚本拉起的 MinIO、Elasticsearch、Milvus Lite。PostgreSQL 留给系统服务。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/volumes"

stop_pid() {
  local file="$1"
  if [[ -f "$file" ]]; then
    local pid
    pid="$(cat "$file")"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" || true
    fi
    rm -f "$file"
  fi
}

stop_pid "$DATA/minio.pid"
stop_pid "$DATA/elasticsearch.pid"
stop_pid "$DATA/milvus.pid"
echo "stopped minio, elasticsearch, milvus"
