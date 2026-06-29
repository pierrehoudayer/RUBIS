"""Results returned by RUBIS deformation solvers."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating]
BoolArray = NDArray[np.bool_]


__all__ = [
    "DeformationResult",
    "RadialResult",
    "SpheroidalResult",
]


@dataclass(kw_only=True)
class DeformationResult:
    """Physical state returned by a stellar deformation solver."""

    zeta: FloatArray
    radial_grid: FloatArray
    cos_theta: FloatArray
    mapping: FloatArray

    density: FloatArray
    pressure: FloatArray

    effective_potential: FloatArray
    effective_potential_derivative: FloatArray

    gravitational_potential_harmonics: FloatArray
    gravitational_potential_derivative_harmonics: FloatArray

    mass: float
    radius: float

    rotation_target: float
    rotation_rate: float

    polar_radius_history: FloatArray
    iterations: int


@dataclass(kw_only=True)
class RadialResult(DeformationResult):
    """Result returned by the radial-coordinate solver."""


@dataclass(kw_only=True)
class SpheroidalResult(DeformationResult):
    """Result returned by the spheroidal-coordinate solver."""

    internal_zeta: FloatArray
    external_zeta: FloatArray
    full_mapping: FloatArray

    internal_mask: BoolArray
    external_mask: BoolArray