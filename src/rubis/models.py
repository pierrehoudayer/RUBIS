"""Data structures describing stellar models."""

import numpy as np
from numpy.typing import NDArray, ArrayLike
from abc import ABC, abstractmethod
from dataclasses import dataclass


__all__ = [
    "PolytropicModelConfig",
    "PolytropeConfig",
    "CompositePolytropeConfig",
    "SphericalModel",
]


@dataclass(kw_only=True)
class PolytropicModelConfig(ABC):
    """Common configuration of a generated polytropic model."""

    radius: float = 1.0
    mass: float = 1.0
    n_points: int = 1001

    @property
    @abstractmethod
    def polytropic_indices(self) -> tuple[float, ...]:
        """Polytropic indices of all regions."""

    @property
    def n_regions(self) -> int:
        return len(self.polytropic_indices)

    @property
    def filename_stem(self) -> str:
        indices = "|".join(f"{index:.1f}" for index in self.polytropic_indices)
        return f"poly_|{indices}|"


@dataclass(kw_only=True)
class PolytropeConfig(PolytropicModelConfig):
    """Configuration of a single polytrope."""

    index: float

    @property
    def polytropic_indices(self) -> tuple[float, ...]:
        return (self.index,)


@dataclass(kw_only=True)
class CompositePolytropeConfig(PolytropicModelConfig):
    """Configuration of a piecewise-polytropic model."""

    indices: ArrayLike
    target_pressures: ArrayLike
    density_jumps: ArrayLike | None = None

    @property
    def polytropic_indices(self) -> tuple[float, ...]:
        if np.ndim(self.indices) == 0:
            return (float(self.indices),)

        return tuple(float(index) for index in self.indices)
    
    
FloatArray = NDArray[np.float64]


@dataclass
class SphericalModel:
    """One-dimensional spherical stellar model."""

    r: FloatArray
    p: FloatArray
    rho: FloatArray
    g: FloatArray

    @property
    def n_points(self) -> int:
        return self.r.size