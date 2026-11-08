#!/usr/bin/env bash
# Run the Kaggle CLI of the project with the correct account.
# Usage: bash jobs/kaggle.sh <arguments of the kaggle command>
# If .kaggle/access_token of the project is not empty, the CLI uses this token.
# With KAGGLE_TOKEN=<name>, the CLI uses .kaggle/access_token_<name> (one more account).
# Otherwise, or with KAGGLE_PERSONAL=1, the CLI uses the personal credentials (~/.kaggle).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOKEN="$ROOT/.kaggle/access_token${KAGGLE_TOKEN:+_$KAGGLE_TOKEN}"
# The value, not the path: a Windows Python does not know the Git Bash paths.
if [ -s "$TOKEN" ] && [ "${KAGGLE_PERSONAL:-0}" != "1" ]; then
  export KAGGLE_API_TOKEN="$(tr -d '[:space:]' < "$TOKEN")"
fi
# The project uses the .venv in the root folder of the repository.
for KAGGLE in "$ROOT/../.venv/Scripts/kaggle.exe" "$ROOT/../.venv/bin/kaggle"; do
  [ -x "$KAGGLE" ] && exec "$KAGGLE" "$@"
done
echo "kaggle.sh: no Kaggle CLI in $ROOT/../.venv" >&2
exit 127
