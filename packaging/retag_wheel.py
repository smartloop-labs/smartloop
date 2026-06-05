#!/usr/bin/env python3
"""Add a PEP 440 local version tag to a built wheel.

Usage: retag_wheel.py <wheel.whl> <variant> <outdir>

Turns e.g. llama_cpp_python-0.3.25-py3-none-linux_x86_64.whl into
llama_cpp_python-0.3.25+<variant>-py3-none-linux_x86_64.whl so CUDA / Metal /
CPU builds can coexist in one Python index and be selected with
`pip install llama-cpp-python==0.3.25+<variant>`. The original wheel is removed.
Requires the `wheel` package (present in a `uv venv --seed` env).
"""
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile


def main() -> None:
    whl, variant, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    if not variant:
        print("retag_wheel: no variant given; leaving wheel unchanged")
        return

    py = sys.executable
    work = tempfile.mkdtemp(prefix="retag-", dir=outdir)
    try:
        subprocess.check_call([py, "-m", "wheel", "unpack", whl, "-d", work])
        src = next(p for p in glob.glob(os.path.join(work, "*")) if os.path.isdir(p))
        dist_info = glob.glob(os.path.join(src, "*.dist-info"))[0]
        meta_path = os.path.join(dist_info, "METADATA")

        text = open(meta_path, encoding="utf-8").read()
        oldv = re.search(r"^Version: (.+)$", text, re.M).group(1).strip()
        if "+" in oldv:
            print(f"retag_wheel: {oldv} already has a local version; skipping")
            return
        newv = f"{oldv}+{variant}"

        open(meta_path, "w", encoding="utf-8").write(
            re.sub(r"^Version: .+$", f"Version: {newv}", text, count=1, flags=re.M)
        )
        name = os.path.basename(dist_info).split("-")[0]
        os.rename(dist_info, os.path.join(src, f"{name}-{newv}.dist-info"))

        # `wheel pack` reads the (edited) .dist-info, so it emits the +variant
        # filename and recomputes RECORD.
        subprocess.check_call([py, "-m", "wheel", "pack", src, "-d", outdir])
        os.remove(whl)
        print(f"retag_wheel: {oldv} -> {newv}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
