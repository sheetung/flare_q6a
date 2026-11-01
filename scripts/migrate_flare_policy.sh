#!/usr/bin/env bash
set -euo pipefail
if (( $# < 2 || $# > 3 )); then
  echo "Usage: $0 model.pt NEW_OUTPUT_DIR [config.yaml|cfgs.pkl]" >&2; exit 2
fi
root="$(cd "$(dirname "$0")/.." && pwd)"
checkpoint="$(realpath -e "$1")"
output="$(realpath -m "$2")"
if (( $# == 3 )); then
  config="$(realpath -e "$3")"
elif [[ -f "$(dirname "$checkpoint")/config.yaml" ]]; then
  config="$(dirname "$checkpoint")/config.yaml"
else
  config="$(realpath -e "$(dirname "$checkpoint")/cfgs.pkl")"
fi
tools_dir="$root/records/run_onnx_export_tools"
[[ ! -e "$output" ]] || { echo "Output already exists: $output" >&2; exit 1; }
[[ -f "$tools_dir/onnx/__init__.py" ]] || { echo "Install ONNX first: scripts/run.sh -m pip install --no-cache-dir --no-deps --target records/run_onnx_export_tools onnx==1.16.2" >&2; exit 1; }
mkdir -p "$(dirname "$output")"
mkdir "$output"
cd "$root"
scripts/run.sh scripts/import_flare_policy.py --checkpoint "$checkpoint" --config "$config" --output "$output/policy.npz"
scripts/run.sh scripts/export_policy_onnx.py --checkpoint "$output/policy.npz" --output "$output/actor.onnx" --onnx-module-dir "$tools_dir"
echo "Migration complete: $output"
