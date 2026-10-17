#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
unset PYTHONPATH PYTHONHOME
export DEV=CL
export RUSTICL_ENABLE="${RUSTICL_ENABLE:-freedreno}"
# Pinned, numerically validated GPU execution profile. Overrides stay explicit.
export RL_PPO_JIT="${RL_PPO_JIT:-1}"
export RL_PPO_JIT_ADAM="${RL_PPO_JIT_ADAM:-1}"
export RL_ROLLOUT_JIT="${RL_ROLLOUT_JIT:-1}"
export RL_PPO_BATCH_REALIZE="${RL_PPO_BATCH_REALIZE:-1}"
export RL_MAIN_CPU="${RL_MAIN_CPU:-auto}"
exec .venv-gpu/bin/python -u "$@"
