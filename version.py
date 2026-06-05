"""Centralized version for the slp CLI package, read from installed metadata."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("slp")
except PackageNotFoundError:
    # Running from a source checkout without an installed dist.
    __version__ = "1.0.0"
