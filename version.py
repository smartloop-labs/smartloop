"""Centralized version for the slp CLI package, read from installed metadata."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("slp")
except PackageNotFoundError:
    # Frozen binary (PyInstaller) or source checkout without an installed
    # dist — no package metadata, so fall back to the package version.
    from smartloop import __version__
