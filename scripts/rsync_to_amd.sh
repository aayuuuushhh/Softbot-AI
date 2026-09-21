# Upload train subset + test to AMD GPU instance
# Usage (from Git Bash or WSL on laptop):
#   bash scripts/rsync_to_amd.sh YOUR_DROPLET_IP

set -euo pipefail
HOST="${1:?Usage: rsync_to_amd.sh DROPLET_IP}"
KEY="${AMD_SSH_KEY:-$HOME/.ssh/id_ed25519_amd}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Prefer a known_hosts entry. Override only for first-time lab setup:
#   AMD_SSH_STRICT=accept-new bash scripts/rsync_to_amd.sh DROPLET_IP
STRICT_MODE="${AMD_SSH_STRICT:-yes}"
echo "Uploading data to root@${HOST}..."
ssh -i "$KEY" -o "StrictHostKeyChecking=${STRICT_MODE}" "root@${HOST}" "mkdir -p /data/train_subset /data/test /results"

rsync -avz --progress -e "ssh -i $KEY" \
  "${REPO_ROOT}/data/train_subset/" "root@${HOST}:/data/train_subset/"

rsync -avz --progress -e "ssh -i $KEY" \
  "${REPO_ROOT}/data/test/" "root@${HOST}:/data/test/"

rsync -avz --progress -e "ssh -i $KEY" \
  "${REPO_ROOT}/ml/" "root@${HOST}:~/DisasterIQ/ml/"

echo "Done. SSH in and run: bash ~/DisasterIQ/ml/finetune/run_amd_pipeline.sh"
