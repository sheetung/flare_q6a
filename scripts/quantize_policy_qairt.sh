#!/usr/bin/env bash
set -euo pipefail
if (( $# != 3 )); then
  echo "Usage: $0 actor.onnx calibration/input_list.txt NEW_OUTPUT_DIR" >&2; exit 2
fi
onnx="$(realpath -e "$1")"
input_list="$(realpath -e "$2")"
output="$(realpath -m "$3")"
[[ ! -e "$output" ]] || { echo "Output already exists: $output" >&2; exit 1; }
for cmd in qairt-converter qairt-quantizer; do
  command -v "$cmd" >/dev/null || { echo "$cmd missing: configure a complete QAIRT SDK first." >&2; exit 1; }
done
python3 - "$input_list" <<'PY'
import math, pathlib, struct, sys
path = pathlib.Path(sys.argv[1])
rows = path.read_text().splitlines()
if not rows:
    raise SystemExit('Calibration input list is empty')
for row in rows:
    file = pathlib.Path(row.strip())
    if not file.is_absolute() or not file.is_file():
        raise SystemExit(f'Expected an absolute raw input path: {row}')
    data = file.read_bytes()
    if len(data) != 26 * 4 or not all(map(math.isfinite, struct.unpack('<26f', data))):
        raise SystemExit(f'Expected 26 finite float32 values: {file}')
PY
mkdir -p "$(dirname "$output")"
mkdir "$output"
qairt-converter --input_network "$onnx" -d observation 1,26 --output_path "$output/actor_fp32.dlc"
qairt-quantizer --input_dlc "$output/actor_fp32.dlc" --input_list "$input_list" --output_dlc "$output/actor_quantized.dlc"
echo "Quantized model: $output/actor_quantized.dlc"
