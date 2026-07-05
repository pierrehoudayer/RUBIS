"""Apply diagnostics and output operations to a converged model."""

from pathlib import Path

import numpy as np

from rubis.api import deform
from rubis.config import (
    DeformationConfig,
    PolytropeConfig,
    RadiativeFluxOptions,
    RotationConfig,
    SolverOptions,
)
from rubis.diagnostics import (
    ModelDiagnostics,
    compute_gravitational_moments,
    compute_virial_balance,
    report_gravitational_moments,
    report_virial_balance,
)
from rubis.flux import compute_radiative_flux
from rubis.io.hdf5 import save_result
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

    model, vacuum, info = deform(config)

    assert vacuum is None

    # Scalar virial balance
    balance = compute_virial_balance(
        model,
        spline_order=config.solver.spline_order,
    )
    report_virial_balance(
        balance,
        verbose=True,
    )

    # Even gravitational moments
    moments = compute_gravitational_moments(
        model,
        max_degree=10,
        spline_order=config.solver.spline_order,
    )
    report_gravitational_moments(
        moments
    )

    # Surface radiative flux
    flux = compute_radiative_flux(
        model,
        RadiativeFluxOptions(
            origin=0.05,
            n_lines=12,
            max_degree=(
                config.solver.max_degree
            ),
            spline_order=(
                config.solver.spline_order
            ),
        ),
    )

    print()
    print(
        "Surface-flux range: "
        f"{np.min(flux.surface_flux):.6f} "
        f"to {np.max(flux.surface_flux):.6f}"
    )

    # Historical RUBIS text output
    Path("models").mkdir(
        exist_ok=True
    )

    save_result(
        "models/polytrope_n3_omega_0p9.h5",
        model,
        vacuum,
        info,
        diagnostics=ModelDiagnostics(
            virial_balance=balance,
            gravitational_moments=moments,
        )
    )


if __name__ == "__main__":
    main()