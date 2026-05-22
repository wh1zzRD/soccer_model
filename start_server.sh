#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-0.0.0.0}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"
export GRADIO_SHARE="${GRADIO_SHARE:-false}"

python3 app.py
