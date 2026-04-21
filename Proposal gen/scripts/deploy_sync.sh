#!/usr/bin/env bash
set -euo pipefail

MODE="inhouse"
REMOTE_HOST="ubuntu@18.136.190.197"
SSH_KEY="${HOME}/Downloads/ai_adoption.pem"
REMOTE_DIR="/srv/apps/proposal-gen"
RESTART_SERVICE="yes"
DRY_RUN="no"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'USAGE'
Usage: scripts/deploy_sync.sh [options]

Options:
  --mode <production|inhouse>   Sync mode. production excludes test/example artifacts.
  --host <user@host>            Remote SSH target (default: ubuntu@18.136.190.197)
  --ssh-key <path>              SSH private key path (default: ~/Downloads/ai_adoption.pem)
  --remote-dir <path>           Remote app directory (default: /srv/apps/proposal-gen)
  --restart <yes|no>            Restart proposal-gen.service after sync (default: yes)
  --dry-run                     Show rsync changes without applying
  --help                        Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --host)
      REMOTE_HOST="${2:-}"
      shift 2
      ;;
    --ssh-key)
      SSH_KEY="${2:-}"
      shift 2
      ;;
    --remote-dir)
      REMOTE_DIR="${2:-}"
      shift 2
      ;;
    --restart)
      RESTART_SERVICE="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="yes"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "${MODE}" != "production" && "${MODE}" != "inhouse" ]]; then
  echo "Invalid --mode '${MODE}'. Use production or inhouse." >&2
  exit 1
fi

if [[ ! -f "${SSH_KEY}" ]]; then
  echo "SSH key not found: ${SSH_KEY}" >&2
  exit 1
fi

if [[ "${RESTART_SERVICE}" != "yes" && "${RESTART_SERVICE}" != "no" ]]; then
  echo "Invalid --restart '${RESTART_SERVICE}'. Use yes or no." >&2
  exit 1
fi

declare -a RSYNC_ARGS=(
  -avz
  --delete
  --exclude '.DS_Store'
  --exclude '.git'
  --exclude '.venv'
  --exclude '__pycache__'
  --exclude '.env'
  --exclude 'generated/'
  --exclude 'app_assets/'
  --exclude 'app_state.db'
  --exclude 'projects.db'
  --exclude 'db.csv'
  --exclude '.osint_cache/'
  --exclude '.research_bundle_cache/'
  --exclude '.chroma/'
  -e "ssh -i ${SSH_KEY}"
)

if [[ "${MODE}" == "production" ]]; then
  RSYNC_ARGS+=(
    --exclude '.env.example'
    --exclude 'internal_api_config.example.json'
    --exclude 'fix.py'
    --exclude 'fix_file.py'
    --exclude 'replace.py'
    --exclude 'tests/'
    --exclude 'test/'
    --exclude 'examples/'
    --exclude '*_test.py'
    --exclude 'test_*.py'
    --exclude '*.example.*'
  )
fi

if [[ "${DRY_RUN}" == "yes" ]]; then
  RSYNC_ARGS+=(--dry-run)
fi

echo "[deploy_sync] mode=${MODE} dry_run=${DRY_RUN}"
echo "[deploy_sync] source=${APP_DIR}/"
echo "[deploy_sync] target=${REMOTE_HOST}:${REMOTE_DIR}/"

rsync "${RSYNC_ARGS[@]}" "${APP_DIR}/" "${REMOTE_HOST}:${REMOTE_DIR}/"

if [[ "${DRY_RUN}" == "yes" ]]; then
  echo "[deploy_sync] dry-run complete. No remote changes were applied."
  exit 0
fi

if [[ "${RESTART_SERVICE}" == "no" ]]; then
  echo "[deploy_sync] sync complete. Service restart skipped."
  exit 0
fi

ssh -i "${SSH_KEY}" "${REMOTE_HOST}" "bash -s" <<EOF
set -euo pipefail
cd "${REMOTE_DIR}"
python3 -m py_compile main/*.py
sudo systemctl daemon-reload
sudo systemctl restart proposal-gen.service
sleep 3
echo "== proposal-gen status =="
sudo systemctl --no-pager --full status proposal-gen.service | sed -n '1,30p'
echo
echo "== health =="
curl -sS -i http://127.0.0.1:5500/health | sed -n '1,12p'
EOF

echo "[deploy_sync] complete."
