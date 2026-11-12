#!/usr/bin/env bash
# Full verification: HVE suite + smoke tests + health check. Exit code is non-zero on any failure.
set -euo pipefail
cd "$(dirname "$0")"
PY=".venv/Scripts/python.exe"
[ -x "$PY" ] || PY=".venv/bin/python"
echo "== HVE + smoke tests =="
"$PY" -m pytest tests/test_hve.py tests/test_smoke.py -v -p no:cacheprovider -W ignore::DeprecationWarning
echo "== health check =="
"$PY" scripts/health_check.py
