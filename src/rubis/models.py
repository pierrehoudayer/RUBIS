"""Data structures describing stellar models."""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from numpy.typing import ArrayLike


__all__ = [
    "PolytropicModelConfig",
    "PolytropeConfig",
    "CompositePolytropeConfig",
]


@dataclass(kw_only=True)
class PolytropicModelConfig(ABC):
    """Common configuration of a generated polytropic model."""

    radius: float = 1.0
    mass: float = 1.0
    resolution: int = 1001

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