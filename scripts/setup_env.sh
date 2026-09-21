#!/usr/bin/env bash
# Create the `uddhar` conda env and install everything.
# torch/torchvision come from the cu124 index and MUST stay paired.
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_NAME=${ENV_NAME:-uddhar}

if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
  echo ">> creating conda env '${ENV_NAME}' (python 3.13)"
  conda create -y -n "${ENV_NAME}" python=3.13
fi

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo ">> installing torch 2.6.0 + torchvision 0.21.0 (cu124)"
pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0 torchvision==0.21.0

echo ">> installing uddhar and its dependencies"
pip install -e ".[dev]"

echo ">> preflight"
python scripts/check_env.py
