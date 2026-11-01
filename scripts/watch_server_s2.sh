#!/usr/bin/env bash
# Reconnect after SSH interruptions; monitor does not start or restart training.
set -uo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
ssh_dir="$root/records/run_ssh_5070ti"
[[ -r "$ssh_dir/id_ed25519" && -r "$ssh_dir/known_hosts" ]] || { echo 'SSH key or known_hosts missing.' >&2; exit 1; }
while true; do
  result=$(ssh -i "$ssh_dir/id_ed25519" -o BatchMode=yes -o StrictHostKeyChecking=yes \
    -o UserKnownHostsFile="$ssh_dir/known_hosts" -o ConnectTimeout=8 \
    -o ServerAliveInterval=10 -o ServerAliveCountMax=2 sheetung@192.168.31.10 \
    'log=/home/sheetung/flare_ws/validation_logs/q6a_transfer_20261101/restart_training.log; status=/home/sheetung/flare_ws/validation_logs/q6a_transfer_20261101/restart_training.exit; date -Is; grep "Learning iteration" "$log" 2>/dev/null | tail -1; if test -f "$status"; then printf "TRAIN_EXIT="; cat "$status"; fi' 2>&1)
  rc=$?
  if (( rc == 0 )); then
    printf '%s\n' "$result"
    if [[ "$result" == *TRAIN_EXIT=0* ]]; then exit 0; fi
    if [[ "$result" == *TRAIN_EXIT=* ]]; then exit 1; fi
  else
    printf 'SSH disconnected; retrying in 15s: %s\n' "$result" >&2
  fi
  sleep 15
done
