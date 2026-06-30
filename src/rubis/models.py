"""Physical model representations used by RUBIS."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .domains import DomainLayout


__all__ = [
    "Model1D",
]


FloatArray = NDArray[np.float64]


@dataclass(kw_only=True)
class Model1D:
    """Normalised one-dimensional model consumed by RUBIS solvers."""

    G: float
    surface_pressure: float

    mass: float
    radius: float

    r: FloatArray
    rho: FloatArray

    domains: DomainLayout
    additional_variables: tuple[FloatArray, ...] = ()

    @property
    def n_points(self) -> int:
        return self.r.size

    @property
    def n_domains(self) -> int:
        return self.domains.n_domains