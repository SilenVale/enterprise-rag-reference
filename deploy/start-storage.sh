#!/usr/bin/env bash
# 拉起本机四个存储：Milvus Lite、Elasticsearch、MinIO、PostgreSQL。
# 已经在听的端口不会再起一份。二进制放在 volumes/，不进 git。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/volumes"
BIN="$DATA/bin"
LOG="$DATA/logs"
mkdir -p "$BIN" "$LOG" "$DATA/minio-data" "$DATA/elasticsearch-data" "$DATA/milvus"

MINIO_RELEASE="RELEASE.2025-09-07T16-13-09Z"
ES_VERSION="8.15.3"

port_open() {
  python3 - "$1" "$2" <<'PY'
import socket, sys
sock = socket.socket()
sock.settimeout(0.4)
code = sock.connect_ex((sys.argv[1], int(sys.argv[2])))
sys.exit(0 if code == 0 else 1)
PY
}

wait_port() {
  local name="$1" host="$2" port="$3"
  for _ in $(seq 1 60); do
    if port_open "$host" "$port"; then
      echo "$name listening on ${host}:${port}"
      return 0
    fi
    sleep 1
  done
  echo "$name did not open ${host}:${port}" >&2
  return 1
}

start_postgres() {
  if port_open 127.0.0.1 5432; then
    echo "postgres already listening"
  elif command -v pg_ctlcluster >/dev/null 2>&1; then
    sudo -n pg_ctlcluster 16 main start
    wait_port postgres 127.0.0.1 5432
  else
    echo "postgres is not listening and pg_ctlcluster is missing" >&2
    exit 1
  fi
  sudo -n -u postgres psql -v ON_ERROR_STOP=1 <<'SQL'
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'rag') THEN
    CREATE ROLE rag LOGIN PASSWORD 'rag-storage';
  END IF;
END
$$;
SQL
  if ! sudo -n -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='rag'" | grep -q 1; then
    sudo -n -u postgres psql -c "CREATE DATABASE rag OWNER rag;"
  fi
}

start_minio() {
  if port_open 127.0.0.1 9000; then
    echo "object storage already listening"
    return
  fi
  if [[ ! -x "$BIN/minio" ]]; then
    curl -fL --retry 3 -o "$BIN/minio" \
      "https://github.com/minio/minio/releases/download/${MINIO_RELEASE}/minio.linux-amd64.${MINIO_RELEASE}"
    chmod +x "$BIN/minio"
  fi
  # 只把服务自己要的两个变量传进进程。不要 export 别的 MINIO_ 前缀。
  nohup env MINIO_ROOT_USER=rag MINIO_ROOT_PASSWORD=rag-storage-secret \
    "$BIN/minio" server --address 127.0.0.1:9000 --console-address 127.0.0.1:9001 \
    "$DATA/minio-data" >"$LOG/minio.log" 2>&1 &
  echo $! >"$DATA/minio.pid"
  wait_port object-storage 127.0.0.1 9000
}

start_elasticsearch() {
  if port_open 127.0.0.1 9200; then
    echo "elasticsearch already listening"
    return
  fi
  if [[ ! -x "$DATA/elasticsearch/bin/elasticsearch" ]]; then
    mkdir -p "$DATA/elasticsearch"
    curl -fL --retry 3 -o "$DATA/elasticsearch.tar.gz" \
      "https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-${ES_VERSION}-linux-x86_64.tar.gz"
    tar -xzf "$DATA/elasticsearch.tar.gz" -C "$DATA/elasticsearch" --strip-components=1
  fi
  cat >"$DATA/elasticsearch/config/elasticsearch.yml" <<EOF
cluster.name: rag-storage
node.name: rag-storage-1
network.host: 127.0.0.1
http.port: 9200
discovery.type: single-node
xpack.security.enabled: false
xpack.security.enrollment.enabled: false
xpack.security.http.ssl.enabled: false
xpack.ml.enabled: false
ingest.geoip.downloader.enabled: false
path.data: ${DATA}/elasticsearch-data
path.logs: ${LOG}/elasticsearch
EOF
  mkdir -p "$LOG/elasticsearch"
  nohup env ES_JAVA_OPTS="-Xms512m -Xmx512m" \
    "$DATA/elasticsearch/bin/elasticsearch" >"$LOG/elasticsearch-stdout.log" 2>&1 &
  echo $! >"$DATA/elasticsearch.pid"
  wait_port elasticsearch 127.0.0.1 9200
}

start_milvus() {
  if port_open 127.0.0.1 19530; then
    echo "milvus already listening"
    return
  fi
  local launcher="$ROOT/.venv/bin/milvus-lite"
  if [[ ! -x "$launcher" ]]; then
    echo "missing ${launcher}; install the project into .venv first" >&2
    exit 1
  fi
  nohup "$launcher" server --data-dir "$DATA/milvus" --host 127.0.0.1 --port 19530 \
    >"$LOG/milvus.log" 2>&1 &
  echo $! >"$DATA/milvus.pid"
  wait_port milvus 127.0.0.1 19530
}

start_postgres
start_minio
start_elasticsearch
start_milvus
echo "storage services are up"
