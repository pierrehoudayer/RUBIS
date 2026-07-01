from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import eval_legendre, roots_legendre

from .numerical import integrate2D
from .rotation import RotationState


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


@dataclass(frozen=True, kw_only=True)
class VirialBalance:
    """
    Contributions to the scalar virial balance.

    Gravitational and thermodynamic contributions are represented
    through their positive-work and thermodynamic-potential conventions.
    """

    kinetic_energy: float
    potential_work: float
    thermodynamic_work: float
    surface_work: float

    @property
    def residual(self) -> float:
        return (
            2.0 * self.kinetic_energy
            - 0.5 * self.potential_work
            - 3.0 * self.thermodynamic_work
            - self.surface_work
        )

    @property
    def scale(self) -> float:
        return (
            2.0 * self.kinetic_energy
            + 0.5 * self.potential_work
            - 3.0 * self.thermodynamic_work
            + self.surface_work
        )

    @property
    def relative_residual(self) -> float:
        return self.residual / self.scale


@dataclass(frozen=True, kw_only=True)
class GravitationalMoments:
    """Even gravitational mass moments of a deformed model."""

    degrees: IntArray
    values: FloatArray
    
    
def compute_virial_balance(
    r2d,
    rho,
    p,
    phi_g2d,
    t,
    rot: RotationState,
    *,
    domains=None,
    spline_order=3,
) -> VirialBalance:
    """
    Compute the scalar virial contributions of a deformed model.

    The gravitational potential must be evaluated on the material
    mapping before calling this representation-independent routine.
    """
    rho2d = rho[:, None]
    omega2d = rot.omega2d(r2d, t)

    potential_work = integrate2D(
        r2d,
        -rho2d * phi_g2d,
        domains=domains,
        k=spline_order,
    )

    kinetic_energy = integrate2D(
        r2d,
        (
            0.5
            * rho2d
            * (1.0 - t[None, :]**2)
            * r2d**2
            * omega2d**2
        ),
        domains=domains,
        k=spline_order,
    )

    thermodynamic_work = -integrate2D(
        r2d,
        p,
        domains=domains,
        k=spline_order,
    )

    _, weights = roots_legendre(t.size)
    surface_work = (
        2.0
        * np.pi
        * (r2d[-1]**3 @ weights)
        * p[-1]
    )

    return VirialBalance(
        kinetic_energy=kinetic_energy,
        potential_work=potential_work,
        thermodynamic_work=thermodynamic_work,
        surface_work=surface_work,
    )
    
    
def compute_gravitational_moments(
    r2d,
    rho,
    t,
    *,
    max_degree=14,
    domains=None,
    spline_order=3,
) -> GravitationalMoments:
    """
    Compute the even gravitational mass moments.

    Multidomain integration may be requested when the material profile
    contains discontinuities.
    """
    degrees = np.arange(0, max_degree + 1, 2)

    values = np.array([
        integrate2D(
            r2d,
            (
                rho[:, None]
                * r2d**l
                * eval_legendre(l, t)
            ),
            domains=domains,
            k=spline_order,
        )
        for l in degrees
    ])

    return GravitationalMoments(
        degrees=degrees,
        values=values,
    )
    
    
def report_virial_balance(
    balance: VirialBalance,
    *,
    verbose=False,
):
    """Display the scalar virial balance."""
    if verbose:
        print(
            f"Kinetic energy     : "
            f"{balance.kinetic_energy:12.10f}"
        )
        print(
            f"Thermodynamic work : "
            f"{balance.thermodynamic_work:12.10f}"
        )
        print(
            f"Potential work     : "
            f"{balance.potential_work:12.10f}"
        )
        print(
            f"Surface work       : "
            f"{balance.surface_work:12.10f}"
        )

    print(
        "Virial theorem verified at "
        f"{round(balance.relative_residual, 16)}"
    )
    
    
def report_gravitational_moments(
    moments: GravitationalMoments,
):
    """Display gravitational mass moments."""
    print(
        "\n+-----------------------+",
        "\n| Gravitational moments |",
        "\n+-----------------------+\n",
    )

    for l, moment in zip(
        moments.degrees,
        moments.values,
    ):
        print(f"Moment n°{l:2d} : {moment:+.10e}")