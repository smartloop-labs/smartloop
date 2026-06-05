#!/bin/sh
# Relocatable launcher for the pre-baked slp venv distribution.
#
# Lives at <dist>/bin/slp next to a bundled standalone CPython at <dist>/python.
# Resolves its own location at runtime so the whole directory can be moved /
# extracted anywhere and still work (no absolute paths baked in).
set -e

# Resolve symlinks so the launcher works when invoked via a symlink
# (e.g. ~/.smartloop/slp -> ~/.smartloop/<version>/bin/slp).
SELF="$0"
while [ -L "$SELF" ]; do
    link_dir="$(CDPATH= cd "$(dirname "$SELF")" && pwd)"
    SELF="$(readlink "$SELF")"
    case "$SELF" in /*) ;; *) SELF="$link_dir/$SELF" ;; esac
done
DIR="$(CDPATH= cd "$(dirname "$SELF")" && pwd)"
PY="$DIR/../python/bin/python3"

SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"

# Expose bundled CUDA / torch shared libraries so llama-cpp & torch can dlopen
# them. No-op on CPU/macOS builds where the nvidia/* dirs are absent.
NVLIBS="$(ls -d "$SITE"/nvidia/*/lib 2>/dev/null | tr '\n' ':')"
export LD_LIBRARY_PATH="${NVLIBS}${SITE}/torch/lib:${LD_LIBRARY_PATH}"

exec "$PY" -m commands.cli "$@"
