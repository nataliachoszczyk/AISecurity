#!/usr/bin/env bash
# Wait until a Kaggle kernel stops, then print its last status.
# Usage: bash wait.sh <owner/kernel-slug>
KAGGLE="$(dirname "$0")/../../.venv/Scripts/kaggle"
while true; do
  status=$("$KAGGLE" kernels status "$1" 2>&1)
  case "$status" in
    *COMPLETE*|*ERROR*|*CANCEL*|*complete*|*error*|*cancel*) echo "$status"; exit 0 ;;
  esac
  sleep 60
done
