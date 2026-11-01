#!/usr/bin/env bash
# Wait for this explicitly started remote run, reconnect, transfer and evaluate.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
ssh_dir="$root/records/run_ssh_5070ti"
remote_run='/home/sheetung/flare_ws/Flare/logs/s2_waypoint_passing_rigid_2026-11-01_11:35:11'
remote_status='/home/sheetung/flare_ws/validation_logs/q6a_transfer_20261101/restart_training.exit'
run_dir="$root/records/run_server_s2_final_20261101"
eval_dir="$root/records/run_server_s2_final_evaluation_20261101"
ssh_args=(-i "$ssh_dir/id_ed25519" -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile="$ssh_dir/known_hosts" -o ConnectTimeout=8 -o ServerAliveInterval=10 -o ServerAliveCountMax=2)
mkdir -p "$run_dir"
exec 9>"$run_dir/.lock"
flock -n 9 || { echo 'Another transfer watcher is running.' >&2; exit 1; }
while :; do
  if state=$(ssh "${ssh_args[@]}" sheetung@192.168.31.10 "if test -f '$remote_status'; then cat '$remote_status'; elif tmux has-session -t q6a_rigid_800 2>/dev/null; then echo RUNNING; else echo MISSING; fi" 2>&1); then
    case "$state" in
      0) break;;
      RUNNING) sleep 15;;
      MISSING) echo 'Remote session disappeared without an exit status.' >&2; exit 1;;
      *) echo "Remote training failed: $state" >&2; exit 1;;
    esac
  else
    printf 'SSH unavailable; retrying in 15 seconds: %s\n' "$state" >&2
    sleep 15
  fi
done
for name in model_800.pt cfgs.pkl; do
  if [[ ! -f "$run_dir/$name" ]]; then
    while :; do
      if ssh "${ssh_args[@]}" sheetung@192.168.31.10 "cat '$remote_run/$name'" > "$run_dir/$name.part"; then
        local_sha=$(sha256sum "$run_dir/$name.part" | cut -d' ' -f1)
        if remote_sha=$(ssh "${ssh_args[@]}" sheetung@192.168.31.10 "sha256sum '$remote_run/$name'" | cut -d' ' -f1) && [[ "$local_sha" == "$remote_sha" ]]; then
          mv "$run_dir/$name.part" "$run_dir/$name"
          break
        fi
      fi
      echo "Transfer or hash check failed for $name; retrying in 15 seconds" >&2
      sleep 15
    done
  fi
done
if [[ ! -f "$run_dir/policy.npz" ]]; then
  scripts/run.sh scripts/import_server_s2.py --checkpoint "$run_dir/model_800.pt" --config "$run_dir/cfgs.pkl" --output "$run_dir/policy.npz" > "$run_dir/import.log" 2>&1
fi
if [[ -e "$eval_dir" ]]; then echo "Evaluation directory already exists: $eval_dir" >&2; exit 1; fi
scripts/run.sh scripts/eval_server_s2.py --checkpoint "$run_dir/policy.npz" --output "$eval_dir" --envs 4 --episodes 4 --steps 1501 > "$run_dir/evaluation.log" 2>&1
printf '%s\n' 'Training transfer and evaluation completed.'
