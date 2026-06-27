"""RUBIS: rotating barotropic stellar models."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rubis")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["__version__"]