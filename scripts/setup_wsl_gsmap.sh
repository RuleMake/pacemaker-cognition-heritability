#!/usr/bin/env bash
# Set up gsMap inside WSL2/Ubuntu.
#
#     bash $CARDIO_ROOT/scripts/setup_wsl_gsmap.sh
#
# WHY NOT THE SYSTEM PYTHON
# -------------------------
# Ubuntu 26.04 ships Python 3.14 and offers no other version in its repos.
# 3.14 is too new for the binary-wheel ecosystem gsMap depends on: pyarrow and
# numcodecs have no cp314 wheels, so pip falls back to building them from source,
# which then fails twice over — pyarrow needs cmake (absent), and numcodecs'
# bundled c-blosc does `typedef _Bool bool;`, illegal now that GCC defaults to C23
# where `bool` is a keyword.
#
# Rather than install cmake and fight each source build, we pin Python 3.12 — the
# version gsMap's own Dockerfile uses — via `uv`. On 3.12 every dependency
# (pyarrow, numcodecs, ncls, sorted_nearest, torch) has a prebuilt manylinux wheel,
# so nothing compiles at all and the install is both faster and far less fragile.
#
# Data stays on the Windows side (/mnt/c/.../Cardio/data) so it is not duplicated,
# but the gsMap reference bundle is copied into the Linux filesystem because gsMap
# hammers it with small reads and /mnt/c is slow for that pattern.

set -euo pipefail

WIN_PROJ="${CARDIO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LIN_PROJ="$HOME/cardio"
VENV="$LIN_PROJ/venv"
PYVER="3.12"

echo "==> [1/7] system packages"
# On a fresh WSL install Ubuntu runs its own apt jobs in the background, which
# hold /var/lib/dpkg/lock-frontend and make any apt-get here fail immediately.
wait_for_apt() {
  local waited=0
  while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 \
     || sudo fuser /var/lib/apt/lists/lock >/dev/null 2>&1; do
    if [ "$waited" -eq 0 ]; then
      echo "    another apt process holds the lock (first-boot auto-update); waiting ..."
    fi
    sleep 10
    waited=$((waited + 10))
    [ $((waited % 60)) -eq 0 ] && echo "    still waiting (${waited}s) ..."
    if [ "$waited" -ge 900 ]; then
      echo "    still locked after 15 min. Inspect with:  ps aux | grep -i apt"
      exit 1
    fi
  done
  [ "$waited" -gt 0 ] && echo "    lock released after ${waited}s"
  return 0
}

wait_for_apt
sudo apt-get update -qq
wait_for_apt
# curl for the uv installer; the rest are only a safety net — with Python 3.12
# every wheel is prebuilt and none of this should actually be needed.
sudo apt-get install -y -qq curl ca-certificates build-essential

echo "==> [2/7] project layout"
mkdir -p "$LIN_PROJ"/{resource,work,results}

echo "==> [3/7] uv + Python $PYVER"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "    uv $(uv --version)"
uv python install "$PYVER"

# A venv built against 3.14 from an earlier attempt is unusable — start clean.
if [ -x "$VENV/bin/python" ] && ! "$VENV/bin/python" -V 2>&1 | grep -q "$PYVER"; then
  echo "    removing stale venv ($("$VENV/bin/python" -V 2>&1))"
  rm -rf "$VENV"
fi
[ -d "$VENV" ] || uv venv --python "$PYVER" "$VENV"
echo "    venv python: $("$VENV/bin/python" -V)"

echo "==> [4/7] torch (CPU) + gsMap — all prebuilt wheels on $PYVER, no compiling"
uv pip install --python "$VENV/bin/python" \
   torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$VENV/bin/python" gsMap

echo "==> [5/7] verify the dependency that blocked the Windows install"
"$VENV/bin/python" - <<'PY'
import pyranges, torch_geometric, torch, pyarrow, numcodecs, sys
print(f"  python          {sys.version.split()[0]}")
print(f"  pyranges        {pyranges.__version__}")
print(f"  torch_geometric {torch_geometric.__version__}")
print(f"  torch           {torch.__version__} | cuda {torch.cuda.is_available()}")
print(f"  pyarrow         {pyarrow.__version__}")
print(f"  numcodecs       {numcodecs.__version__}")
PY
echo "    gsmap CLI: $("$VENV/bin/gsmap" --help >/dev/null 2>&1 && echo OK || echo FAILED)"

echo "==> [6/7] reference bundle -> Linux filesystem"
if [ ! -d "$LIN_PROJ/resource/gsMap_resource" ]; then
  cp "$WIN_PROJ/data/resource/gsMap_resource.tar.gz" "$LIN_PROJ/resource/"
  tar xzf "$LIN_PROJ/resource/gsMap_resource.tar.gz" -C "$LIN_PROJ/resource/"
  rm -f "$LIN_PROJ/resource/gsMap_resource.tar.gz"
fi
du -sh "$LIN_PROJ/resource"
find "$LIN_PROJ/resource" -maxdepth 2 -type d | head -12

echo "==> [7/7] link the Windows-side data"
ln -sfn "$WIN_PROJ/data" "$LIN_PROJ/data_win"
ls "$LIN_PROJ/data_win/"

cat <<EOF

================================================================
gsMap environment ready.

  activate :  source $VENV/bin/activate
  data     :  $LIN_PROJ/data_win/   (-> Windows side, not duplicated)
  reference:  $LIN_PROJ/resource/gsMap_resource
  work/out :  $LIN_PROJ/work  $LIN_PROJ/results

Next:
  bash "$WIN_PROJ/scripts/20_run_gsmap.sh"
================================================================
EOF
