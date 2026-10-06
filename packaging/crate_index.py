#!/usr/bin/env python3
"""Add the packaged crate to a sparse registry index file.

The smartloop crate is published to a static Cargo registry under
dl.smartloop.ai/crates/ (see the publish job in .github/workflows/build.yml),
not to crates.io. A sparse registry is plain files: config.json, one index
file per crate (one JSON line per version) and the .crate archives. This
writes the index line for the crate `cargo package` just built, replacing any
line for the same version so retags re-publish cleanly.

    python3 packaging/crate_index.py target/package/smartloop-1.2.8.crate registry/

writes registry/sm/ar/smartloop and prints that path relative to registry/.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def index_path(name: str) -> str:
    """The index file's path relative to the registry root."""
    name = name.lower()
    if len(name) <= 2:
        return f"{len(name)}/{name}"
    if len(name) == 3:
        return f"3/{name[0]}/{name}"
    return f"{name[:2]}/{name[2:4]}/{name}"


def dependency(dep: dict) -> dict:
    source = dep.get("source") or ""
    if not source.startswith("registry+"):
        sys.exit(f"{dep['name']}: only registry dependencies can be published ({source or 'path'})")
    entry = {
        "name": dep["rename"] or dep["name"],
        "req": dep["req"],
        "features": dep["features"],
        "optional": dep["optional"],
        "default_features": dep["uses_default_features"],
        "target": dep["target"],
        "kind": dep["kind"] or "normal",
        # Omitted means "this registry"; every dependency lives on crates.io.
        "registry": source.removeprefix("registry+"),
    }
    if dep["rename"]:
        entry["package"] = dep["name"]
    return entry


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    crate, root = Path(sys.argv[1]), Path(sys.argv[2])

    metadata = json.loads(
        subprocess.run(
            ["cargo", "metadata", "--no-deps", "--format-version", "1", "--locked"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    pkg = next(p for p in metadata["packages"] if p["name"] == "smartloop")

    if crate.name != f"{pkg['name']}-{pkg['version']}.crate":
        sys.exit(f"{crate.name} doesn't match {pkg['name']} {pkg['version']}")

    entry = {
        "name": pkg["name"],
        "vers": pkg["version"],
        "deps": [dependency(d) for d in pkg["dependencies"]],
        "cksum": hashlib.sha256(crate.read_bytes()).hexdigest(),
        "features": pkg["features"],
        "yanked": False,
        "links": pkg["links"],
        "rust_version": pkg["rust_version"],
    }

    index = root / index_path(pkg["name"])
    lines = []
    if index.exists():
        lines = [
            line
            for line in index.read_text().splitlines()
            if line.strip() and json.loads(line)["vers"] != entry["vers"]
        ]
    lines.append(json.dumps(entry, separators=(",", ":")))
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text("\n".join(lines) + "\n")
    print(f"{pkg['name']} {pkg['version']}: {len(lines)} version(s) in the index", file=sys.stderr)
    print(index_path(pkg["name"]))


if __name__ == "__main__":
    main()
