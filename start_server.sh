#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -z "${VIRTUAL_ENV:-}" && -d ".venv" ]]; then
  # Prefer the project venv when the script is started from a clean shell.
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  nvidia_site="$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia"
  if [[ -d "$nvidia_site/cublas/lib" ]]; then
    cuda_libs=(
      "$nvidia_site/cublas/lib"
      "$nvidia_site/cuda_runtime/lib"
      "$nvidia_site/cudnn/lib"
      "$nvidia_site/cufft/lib"
      "$nvidia_site/curand/lib"
      "$nvidia_site/cusolver/lib"
      "$nvidia_site/cusparse/lib"
      "$nvidia_site/nccl/lib"
    )
    cuda_library_path="$(IFS=:; echo "${cuda_libs[*]}")"
    export LD_LIBRARY_PATH="$cuda_library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LD_PRELOAD="$nvidia_site/cublas/lib/libcublasLt.so.12:$nvidia_site/cublas/lib/libcublas.so.12${LD_PRELOAD:+:$LD_PRELOAD}"
  fi
fi

export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-0.0.0.0}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"
export GRADIO_SHARE="${GRADIO_SHARE:-false}"

python3 app.py
