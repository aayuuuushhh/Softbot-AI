#!/usr/bin/env bash
# Apply DisasterIQ patches to vendored michal2409/xView2 (see patch_pytorch_xview2.py)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
python3 "$SCRIPT_DIR/patch_pytorch_xview2.py"
