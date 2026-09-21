#!/usr/bin/env bash
# Start a local mongod against a repo-local data directory.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .data/db .data/log
exec mongod --dbpath .data/db --logpath .data/log/mongod.log --port 27017 "$@"
