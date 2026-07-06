"""RUBIS: rotating barotropic stellar and planetary models."""

from importlib.metadata import PackageNotFoundError, version

from .api import deform
from .config import (
    CompositePolytropeConfig,
    DeformationConfig,
    HDF5ModelConfig,
    LegacyModelConfig,
    PolytropeConfig,
    RadiativeFluxOptions,
    RotationConfig,
    SolverOptions,
)
from .models import Model2D, VacuumModel2D
from .results import SolverInfo, SolverOutput


try:
    __version__ = version("rubis")
except PackageNotFoundError:
    __version__ = "unknown"


__all__ = [
    "CompositePolytropeConfig",
    "DeformationConfig",
    "HDF5ModelConfig",
    "LegacyModelConfig",
    "Model2D",
    "PolytropeConfig",
    "RadiativeFluxOptions",
    "RotationConfig",
    "SolverInfo",
    "SolverOptions",
    "SolverOutput",
    "VacuumModel2D",
    "__version__",
    "deform",
]
