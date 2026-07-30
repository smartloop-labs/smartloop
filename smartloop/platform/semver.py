"""Minimal semver parsing/comparison for release update checks.

Mirrors the ordering rules the cloud (slp-api) uses when picking the latest
published release: compare on (major, minor, patch), and a release WITHOUT a
prerelease sorts above the same version WITH one (1.0.15 > 1.0.15-beta.1).

Kept intentionally small — no third-party dependency.
"""

import re
from typing import Optional

# x.y.z with an optional -prerelease suffix (build metadata after '+' is ignored)
_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?$")


def _prerelease_key(prerelease: Optional[str]) -> tuple:
    """Sort key for the prerelease portion.

    A missing prerelease outranks any present one, so we prefix with 1 for
    released versions and 0 for prereleases.  Within a prerelease, identifiers
    are compared numerically when numeric, else lexically (per semver §11).
    """
    if not prerelease:
        return (1,)
    ids: list = []
    for ident in prerelease.split("."):
        if ident.isdigit():
            ids.append((0, int(ident)))
        else:
            ids.append((1, ident))
    return (0, tuple(ids))


def parse(version: Optional[str]) -> Optional[tuple]:
    """Return a sortable key for *version*, or ``None`` if it isn't valid semver."""
    if not version:
        return None
    m = _SEMVER_RE.match(version.strip())
    if not m:
        return None
    major, minor, patch, prerelease = m.groups()
    return (int(major), int(minor), int(patch), _prerelease_key(prerelease))


def is_newer(candidate: str, current: str) -> bool:
    """Return True when *candidate* is a strictly newer semver than *current*.

    If either value can't be parsed, returns False (fail closed — don't claim
    an update is available on garbage input).
    """
    c = parse(candidate)
    cur = parse(current)
    if c is None or cur is None:
        return False
    return c > cur
