"""Physical model representations used by RUBIS."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .domains import DomainLayout


__all__ = [
    "Model1D",
    "Model2D",
    "VacuumModel2D",
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


@dataclass(kw_only=True)
class Model2D:
    """
    Converged two-dimensional material model.

    Thermodynamic and effective-potential profiles are defined on zeta.
    Gravitational, centrifugal, and rotation fields are evaluated on
    the material mapping r2d(zeta, t). Every derivative ending in `_z`
    is taken with respect to zeta at fixed t.
    """

    G: float
    surface_pressure: float
    mass: float
    radius: float
    omega_eq: float

    zeta: FloatArray
    t: FloatArray
    r2d: FloatArray

    rho: FloatArray
    p: FloatArray
    additional_variables: tuple[FloatArray, ...]

    phi_eff: FloatArray
    phi_eff_z: FloatArray

    phi_g: FloatArray
    phi_g_z: FloatArray

    phi_c: FloatArray
    phi_c_z: FloatArray

    omega: FloatArray

    domains: DomainLayout

    @property
    def n_points(self) -> int:
        return self.zeta.size

    @property
    def angular_resolution(self) -> int:
        return self.t.size

    @property
    def n_domains(self) -> int:
        return self.domains.n_domains


@dataclass(kw_only=True)
class VacuumModel2D:
    """
    Two-dimensional model of an exterior vacuum domain.

    All potential and rotation fields are evaluated on r2d(zeta, t).
    Every derivative ending in `_z` is taken with respect to zeta at
    fixed t.
    """

    G: float
    mass: float
    radius: float
    omega_eq: float

    zeta: FloatArray
    t: FloatArray
    r2d: FloatArray

    phi_g: FloatArray
    phi_g_z: FloatArray

    phi_c: FloatArray
    phi_c_z: FloatArray

    phi_eff: FloatArray
    phi_eff_z: FloatArray

    omega: FloatArray

    domains: DomainLayout

    @property
    def n_points(self) -> int:
        return self.zeta.size

    @property
    def angular_resolution(self) -> int:
        return self.t.size

    @property
    def n_domains(self) -> int:
        return self.domains.n_domains