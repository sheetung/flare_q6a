#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
unset PYTHONPATH PYTHONHOME
export DEV=CL
export RUSTICL_ENABLE="${RUSTICL_ENABLE:-freedreno}"
exec .venv-gpu/bin/python -u "$@"
