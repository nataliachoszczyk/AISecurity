#!/usr/bin/env bash
# Follow a Kaggle kernel until it stops. Write each status change into JOBS.md.
# Usage: bash wait.sh <owner/kernel-slug> "<job name in the JOBS.md row>"
DIR="$(dirname "$0")"
KAGGLE="$DIR/../../.venv/Scripts/kaggle"
[ -x "$KAGGLE" ] || [ -x "$KAGGLE.exe" ] || KAGGLE="$DIR/../../.venv/bin/kaggle"  # Linux and macOS
last=""
while true; do
  raw=$("$KAGGLE" kernels status "$1" 2>&1)
  case "$raw" in
    *QUEUED*) status="queued" ;;
    *RUNNING*) status="running" ;;
    *COMPLETE*) status="complete" ;;
    *ERROR*) status="failed" ;;
    *CANCEL*) status="cancelled" ;;
    *) status="$last" ;;
  esac
  if [ -n "$status" ] && [ "$status" != "$last" ]; then
    sed -i "/^| [0-9-]* | $2 |/ s/| [a-z ]* |\$/| $status |/" "$DIR/JOBS.md"
    echo "$(date +%H:%M) $status"
    last="$status"
  fi
  case "$status" in complete|failed|cancelled) echo "$raw"; exit 0 ;; esac
  sleep 60
done
