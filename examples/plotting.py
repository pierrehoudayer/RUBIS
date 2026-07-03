"""Visualise a converged model and its radiative flux."""

import matplotlib.pyplot as plt

from rubis.api import deform
from rubis.config import (
    DeformationConfig,
    PolytropeConfig,
    RadiativeFluxOptions,
    RotationConfig,
    SolverOptions,
)
from rubis.flux import compute_radiative_flux
from rubis.plotting import (
    plot_gravitational_harmonics,
    plot_model_field,
    plot_radiative_flux_lines,
    plot_radiative_flux_surface,
)
from rubis.rotation_profiles import solid


def main():
    config = DeformationConfig(
        model=PolytropeConfig(
            index=3.0,
            n_points=501,
        ),
        rotation=RotationConfig(
            profile=solid,
            target=0.9,
        ),
        solver=SolverOptions(
            max_degree=51,
            angular_resolution=51,
            full_rate=1,
            mapping_precision=1.0e-10,
            spline_order=5,
            lagrange_order=3,
            verbose=True,
        ),
    )

    model, vacuum, _ = deform(config)

    plot_model_field(
        model,
        model.rho,
        vacuum=vacuum,
        max_degree=config.solver.max_degree,
        label=r"$\rho$",
        show_surfaces=True,
        surface_count=30,
    )

    plot_gravitational_harmonics(
        model,
        vacuum=vacuum,
        max_degree=config.solver.max_degree,
    )

    flux = compute_radiative_flux(
        model,
        RadiativeFluxOptions(
            origin=0.05,
            n_lines=12,
            max_degree=config.solver.max_degree,
            spline_order=config.solver.spline_order,
        ),
    )

    plot_radiative_flux_lines(flux)

    plot_radiative_flux_surface(
        model,
        flux,
        cmap="stellar_r",
    )

    plt.show()


if __name__ == "__main__":
    main()