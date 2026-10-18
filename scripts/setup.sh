#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -m)" != aarch64 ]]; then
  echo 'This lockfile is tested on Linux aarch64 Qualcomm boards.' >&2; exit 1
fi
PYTHON="${RL_PYTHON:-python3.12}"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,12), "Use Python 3.12 for the tested environment"'
"$PYTHON" -c 'import platform; name,version=platform.libc_ver(); assert name == "glibc" and tuple(map(int,version.split(".")[:2])) >= (2,38), "This lockfile needs glibc >= 2.38 (Ubuntu 24.04 recommended)"'
bash scripts/fetch_flare.sh
for source in Genesis/setup.py rsl_rl_reference/setup.py; do
  [[ -f "$source" ]] || { echo 'Run git submodule update --init --recursive first.' >&2; exit 1; }
done
# Do not inherit ROS/system PYTHONPATH into the reproducible training environment.
unset PYTHONPATH PYTHONHOME
"$PYTHON" -m venv .venv-gpu
.venv-gpu/bin/python -m pip install -r requirements-q6a.lock
.venv-gpu/bin/python -m pip install -e ./Genesis --no-deps
printf '%s\n' 'Environment ready. Next: scripts/run.sh flare_q6a/tests/check_parity.py'
