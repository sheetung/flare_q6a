#!/usr/bin/env bash
# Download/extract only; never install packages or change system ICDs.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ "$(dpkg --print-architecture)" == arm64 ]] || { echo 'This runtime is pinned for Linux arm64.' >&2; exit 1; }
probe_root="${RL_QCOM_OPENCL_ROOT:-$PWD/records/run_opencl_driver_probe}"
mkdir -p "$probe_root/packages" "$probe_root/root" "$probe_root/icd"
packages=(
  'qcom-adreno-cl1=1.855.3+rev3+repack1-0ubuntu1'
  'qcom-adreno-common1=1.855.3+rev3+repack1-0ubuntu1'
  'qcom-libdmabufheap=1.1.0+250131+rel1.0+nmu1'
  'qcom-property-vault=1.1.0-0ubuntu1'
)
(cd "$probe_root/packages" && apt-get download "${packages[@]}")
for spec in "${packages[@]}"; do
  name="${spec%%=*}"; version="${spec#*=}"
  dpkg-deb -x "$probe_root/packages/${name}_${version}_arm64.deb" "$probe_root/root"
done
printf '%s\n' "$probe_root/root/usr/lib/aarch64-linux-gnu/libOpenCL_adreno.so.1" > "$probe_root/icd/adrenocl.icd"
python3 - "$probe_root" <<'PY'
import hashlib,json,sys,subprocess
from pathlib import Path
root=Path(sys.argv[1]);packages=[]
for p in sorted((root/'packages').glob('*.deb')):
    identity=subprocess.check_output(['dpkg-deb','-f',str(p),'Package','Version'],text=True)
    packages.append({'file':p.name,'identity':identity.strip(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(root/'packages.json').write_text(json.dumps({'method':'download and extract; no system installation','packages':packages},indent=2)+'\n')
PY
printf 'Isolated Qualcomm OpenCL runtime: %s\n' "$probe_root"
