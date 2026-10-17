#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -m)" != aarch64 ]]; then
  echo 'This lockfile is tested on Linux aarch64 Qualcomm boards.' >&2; exit 1
fi
PYTHON="${RL_PYTHON:-python3.12}"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,12), "Use Python 3.12 for the tested environment"'
for source in Flare/README.md Genesis/setup.py rsl_rl_reference/setup.py; do
  [[ -f "$source" ]] || { echo 'Run git submodule update --init --recursive first.' >&2; exit 1; }
done
# Do not inherit ROS/system PYTHONPATH into the reproducible training environment.
unset PYTHONPATH PYTHONHOME
"$PYTHON" -m venv .venv-gpu
.venv-gpu/bin/python -m pip install -r requirements-q6a.lock
.venv-gpu/bin/python -m pip install -e ./Genesis --no-deps
printf '%s\n' 'Environment ready. Next: scripts/run.sh flare_q6a/check_parity.py'
