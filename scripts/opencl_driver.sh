#!/usr/bin/env bash
# Select one OpenCL driver for this process; system ICDs remain unchanged.
set -euo pipefail
cd "$(dirname "$0")/.."
backend="${1:?Usage: bash scripts/opencl_driver.sh rusticl|qcom SCRIPT [ARGS...]}"
shift
probe_root="${RL_QCOM_OPENCL_ROOT:-$PWD/records/run_opencl_driver_probe}"
export RL_OPENCL_DRIVER="$backend"
export OPENCL_PATH=/lib/aarch64-linux-gnu/libOpenCL.so.1
case "$backend" in
  rusticl)
    export OCL_ICD_VENDORS=/etc/OpenCL/vendors/rusticl.icd
    ;;
  qcom)
    vendor_lib="$probe_root/root/usr/lib/aarch64-linux-gnu"
    [[ -f "$vendor_lib/libOpenCL_adreno.so.1" ]] || { echo 'Prepare the isolated Qualcomm runtime first; see docs/OPENCL_COMPARISON.md.' >&2; exit 1; }
    export LD_LIBRARY_PATH="$vendor_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export OCL_ICD_VENDORS="$probe_root/icd"
    ;;
  *) echo "Unknown OpenCL driver: $backend" >&2; exit 2 ;;
esac
exec bash scripts/run.sh "$@"
