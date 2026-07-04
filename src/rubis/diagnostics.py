from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import eval_legendre, roots_legendre

from .models import Model2D
from .quadrature import integrate_axisymmetric


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
    model: Model2D,
    *,
    spline_order=3,
) -> VirialBalance:
    """
    Compute the scalar virial contributions of a material model.

    Every integral is evaluated independently in each material domain,
    so discontinuous density profiles require no additional information
    from the solver.
    """
    domains = model.domains.domain_ranges
    rho2d = model.rho[:, None]

    potential_work = integrate_axisymmetric(
        model.r2d,
        -rho2d * model.phi_g,
        domains=domains,
    )

    kinetic_energy = integrate_axisymmetric(
        model.r2d,
        (
            0.5
            * rho2d
            * (1.0 - model.t[None, :]**2)
            * model.r2d**2
            * model.omega**2
        ),
        domains=domains,
    )

    thermodynamic_work = -integrate_axisymmetric(
        model.r2d,
        model.p,
        domains=domains,
    )

    _, weights = roots_legendre(model.angular_resolution)

    surface_work = (
        2.0
        * np.pi
        * (model.r2d[-1]**3 @ weights)
        * model.surface_pressure
    )

    return VirialBalance(
        kinetic_energy=kinetic_energy,
        potential_work=potential_work,
        thermodynamic_work=thermodynamic_work,
        surface_work=surface_work,
    )


def compute_gravitational_moments(
    model: Model2D,
    *,
    max_degree=14,
    spline_order=3,
) -> GravitationalMoments:
    """
    Compute the even gravitational mass moments.

    Material interfaces are integrated domain by domain according to
    the layout stored by the model.
    """
    degrees = np.arange(0, max_degree + 1, 2)

    values = np.array([
        integrate_axisymmetric(
            model.r2d,
            (
                model.rho[:, None]
                * model.r2d**l
                * eval_legendre(l, model.t)
            ),
            domains=model.domains.domain_ranges,
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