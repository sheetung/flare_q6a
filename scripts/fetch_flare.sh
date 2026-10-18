#!/usr/bin/env bash
# Fetch the runtime reference without adding it to the parent Git repository.
set -euo pipefail
cd "$(dirname "$0")/.."
revision=41f03b94f2f724205e5e24617010467aae2a3f98
if [[ ! -e Flare ]]; then
  git clone --no-checkout https://github.com/BEI11HAI/Flare.git Flare
  git -C Flare checkout --detach "$revision"
fi
actual=$(git -C Flare rev-parse HEAD)
if [[ "$actual" != "$revision" ]]; then
  echo "Flare revision mismatch: expected $revision, found $actual. Existing files were preserved." >&2
  exit 1
fi
for source in envs/scenario1_env.py utils/mixer.py assets/suspended_system/real_sus_exp.urdf; do
  [[ -f "Flare/$source" ]] || { echo "Missing FLARE runtime dependency: $source" >&2; exit 1; }
done
printf 'External FLARE ready: %s\n' "$actual"
