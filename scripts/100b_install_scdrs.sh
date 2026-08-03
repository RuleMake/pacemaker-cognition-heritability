#!/usr/bin/env bash
# Install scdrs into the gsMap venv.
#
# The venv was built by `uv`, which does not place a `pip` shim inside it, so the
# obvious `pip install` fails with "command not found". Two routes are tried in order:
# ensurepip to bootstrap pip inside the venv, then uv targeting the venv interpreter.
set -uo pipefail
VP="$HOME/cardio/venv/bin/python"

"$VP" -m ensurepip --upgrade >/dev/null 2>&1 || true
if "$VP" -m pip --version >/dev/null 2>&1; then
  echo "using pip inside the venv"
  "$VP" -m pip install --upgrade-strategy only-if-needed scdrs 2>&1 | tail -12
elif command -v uv >/dev/null 2>&1; then
  echo "using uv"
  uv pip install --python "$VP" scdrs 2>&1 | tail -12
else
  echo "neither pip nor uv available" >&2
  exit 1
fi

echo
"$VP" - <<'PY'
import scdrs
import scanpy
import anndata
print("scdrs   ", scdrs.__version__)
print("scanpy  ", scanpy.__version__)
print("anndata ", anndata.__version__)
PY
