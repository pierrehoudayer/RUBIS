"""Poisson-operator couplings in spheroidal coordinates."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .legendre import Legendre_coupling
from .mapping import ExtendedMappingDerivatives


__all__ = [
    "PoissonCouplings",
    "compute_poisson_couplings",
]


FloatArray = NDArray[np.floating]


@dataclass(kw_only=True)
class PoissonCouplings:
    """Legendre-mode couplings of the spheroidal Poisson operator."""

    zz: FloatArray
    zt: FloatArray
    tt: FloatArray
    boundary: FloatArray
    
    
def compute_poisson_couplings(
    r2d,
    der: ExtendedMappingDerivatives,
    t,
    max_degree,
    alpha=2,
) -> PoissonCouplings:
    """Compute the Legendre couplings of the Poisson operator.

    Parameters
    ----------
    r2d : ndarray, shape (N, M)
        Mapping including the external vacuum domain.
    der : ExtendedMappingDerivatives
        Available derivatives of the extended mapping.
    t : ndarray, shape (M,)
        Angular coordinate, with t = cos(theta).
    max_degree : int
        Number of Legendre degrees retained in the projection.
    alpha : float, optional
        Coefficient of the mixed angular derivative term.

    Returns
    -------
    PoissonCouplings
        Coupling matrices between the retained even Legendre modes.
    """
    q = 1 - t**2
    l = np.arange(0, max_degree, 2)

    zz = Legendre_coupling(
        (r2d**2 + q * der.r_t**2) / der.r_z,
        max_degree,
        der=(0, 0),
    )

    zt = (
        Legendre_coupling(
            q * der.r_tt - 2 * t * der.r_t,
            max_degree,
            der=(0, 0),
        )
        + alpha * Legendre_coupling(
            q * der.r_t,
            max_degree,
            der=(0, 1),
        )
    )

    tt = (
        Legendre_coupling(
            der.r_z,
            max_degree,
            der=(0, 0),
        )
        * l * (l + 1)
    )

    boundary = Legendre_coupling(
        1 / der.r_z,
        max_degree,
        der=(0, 0),
    )

    return PoissonCouplings(
        zz=zz,
        zt=zt,
        tt=tt,
        boundary=boundary,
    )