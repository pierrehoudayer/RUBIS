"""Radiative-flux post-processing for converged models."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.special import roots_legendre

from .config import RadiativeFluxOptions
from .legendre import (
    pl_eval_2D,
    pl_project_2D,
)
from .mapping import (
    compute_mapping_derivatives,
    compute_mapping_geometry,
)
from .models import Model2D
from .interpolation import interpolate_func
from .quadrature import integrate


FloatArray = NDArray[np.float64]


__all__ = [
    "RadiativeFlux",
    "compute_radiative_flux",
]


@dataclass(frozen=True, kw_only=True)
class RadiativeFlux:
    """
    Radiative-flux reconstruction on a converged model.

    Surface values are sampled on `surface_t` and represented by their
    Legendre coefficients `surface_flux_l`. Characteristic lines are
    stored on `line_zeta`.
    """

    line_zeta: FloatArray
    line_r: FloatArray
    line_t: FloatArray

    surface_t: FloatArray
    surface_flux: FloatArray
    surface_flux_l: FloatArray

    max_degree: int
    
    
def compute_radiative_flux(
    model: Model2D,
    options: RadiativeFluxOptions = RadiativeFluxOptions(),
) -> RadiativeFlux:
    """
    Reconstruct the normalized surface radiative flux.

    Characteristics are integrated inward from the surface to the
    specified origin. Flux conservation along each characteristic
    determines the surface distribution.
    """
    if model.n_domains > 1:
        raise ValueError(
            "Radiative-flux reconstruction currently requires "
            "a single-domain model."
        )

    zeta = model.zeta
    r2d = model.r2d
    t_grid = model.t

    L = (
        model.angular_resolution
        if options.max_degree is None
        else options.max_degree
    )
    spl_order = options.spline_order
    z0 = options.origin
    j_lines = options.n_lines

    # Initial angular positions of the downward characteristics
    surface_t, surface_weights = roots_legendre(2 * j_lines)
    t_dw = surface_t[:j_lines]
    weights_dw = surface_weights[:j_lines]

    # Restrict the mapping to the flux domain
    flux_domain = zeta >= z0
    z = zeta[flux_domain]
    r2d_flux = r2d[flux_domain]

    # Integration coordinate measured inward from the surface
    depth = (1.0 - z)[::-1]

    # Metric terms
    der = compute_mapping_derivatives(
        r2d_flux,
        z,
        t_grid,
        max_degree=L,
        spline_order=spl_order,
        domain_ranges=(slice(None),),
    )
    geo = compute_mapping_geometry(
        r2d_flux,
        der,
        t_grid,
    )

    r_l   = pl_project_2D(r2d_flux, L)
    rhs_l = pl_project_2D(geo.gg, L, even=False)
    jac_l = pl_project_2D(geo.jacobian, L)

    # Characteristic equation dt / d(depth)
    def flux_line_rhs(depth_eval, t_eval):
        rhs_t = np.atleast_2d(
            pl_eval_2D(rhs_l, t_eval.ravel())
        )

        return np.array([
            interpolate_func(
                depth,
                -rhs_tk[::-1],
                k=3,
            )(depth_eval)
            for rhs_tk in rhs_t.T
        ]).reshape(t_eval.shape)

    def flux_line_jacobian(depth_eval, t_eval):
        jac_t = np.atleast_2d(
            pl_eval_2D(jac_l, t_eval.ravel())
        )

        jacobian = np.diag([
            interpolate_func(depth, -jac_tk[::-1], k=3)(depth_eval)
            for jac_tk in jac_t.T
        ])

        return jacobian.reshape(-1, *t_eval.shape)

    # Integrate the characteristics from the surface to z0
    solution = solve_ivp(
        fun=flux_line_rhs,
        t_span=(
            0.0,
            1.0 - z0,
        ),
        y0=t_dw,
        method="LSODA",
        dense_output=True,
        rtol=1.0e-4,
        atol=1.0e-4,
        jac=flux_line_jacobian,
        vectorized=True,
    )

    t_lines = solution.sol(depth).T[::-1]

    # Mapping and angular derivative along the characteristics
    r_lines, r_t_lines = np.moveaxis(
        np.array([
            pl_eval_2D(r_l[i], t_i, der=1)
            for i, t_i in enumerate(t_lines)
        ]),
        0,
        1,
    )

    # Radial derivative and relative divergence
    r_z_l = pl_project_2D(der.r_z, L)
    divrel_z_l = pl_project_2D(geo.divrelz, L)

    r_z_lines = np.array([
        pl_eval_2D(r_z_l[i], t_i)
        for i, t_i in enumerate(t_lines)
    ])
    divrel_z_lines = np.array([
        pl_eval_2D(divrel_z_l[i], t_i)
        for i, t_i in enumerate(t_lines)
    ])

    # Flux transport along each characteristic
    Q_z = np.exp([
        -integrate(z, divrel_z_i)
        for divrel_z_i in divrel_z_lines.T
    ])

    surface_factor = (
        r_lines[-1] ** 2
        + (1.0 - t_dw**2)
        * r_t_lines[-1] ** 2
    ) / r_z_lines[-1]

    Q0 = 1.0 / np.sum(Q_z * surface_factor * weights_dw)

    gzz_surface_l = pl_project_2D(geo.gzz[-1], L)
    gzz_surface = pl_eval_2D(gzz_surface_l, t_lines[-1])

    Q_dw = Q0 * Q_z * np.sqrt(np.abs(gzz_surface))

    # Restore equatorial symmetry
    surface_flux = np.hstack((
        Q_dw,
        Q_dw[::-1],
    ))
    surface_flux_l = pl_project_2D(
        surface_flux,
        2 * j_lines,
    )

    line_r = np.hstack((
        r_lines,
        r_lines[:, ::-1],
    ))
    line_t = np.hstack((
         t_lines,
        -t_lines[:, ::-1],
    ))

    return RadiativeFlux(
        line_zeta=z.copy(),
        line_r=line_r,
        line_t=line_t,
        surface_t=surface_t,
        surface_flux=surface_flux,
        surface_flux_l=surface_flux_l,
        max_degree=L,
    )