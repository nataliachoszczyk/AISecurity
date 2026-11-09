#!/usr/bin/env bash
# Run the Kaggle CLI of the project with the correct account.
# Usage: bash jobs/kaggle.sh <arguments of the kaggle command>
# If .kaggle/access_token of the project is not empty, the CLI uses this token.
# Otherwise the CLI uses the personal credentials (~/.kaggle/access_token).
set -euo pipefail
# The root folder of the repository is two levels above this script.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TOKEN="$ROOT/.kaggle/access_token"
# The value, not the path: a Windows Python does not know the Git Bash paths.
if [ -s "$TOKEN" ]; then
  export KAGGLE_API_TOKEN="$(tr -d '[:space:]' < "$TOKEN")"
fi
for KAGGLE in "$ROOT/.venv/Scripts/kaggle.exe" "$ROOT/.venv/bin/kaggle"; do
  [ -x "$KAGGLE" ] && exec "$KAGGLE" "$@"
done
echo "kaggle.sh: no Kaggle CLI in $ROOT/.venv" >&2
exit 127
