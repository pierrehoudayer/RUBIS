import numpy as np

from rubis.api import deform
from rubis.config import (
    DeformationConfig,
    RotationConfig,
    SolverOptions,
)
from rubis.models import CompositePolytropeConfig
from rubis.options import (
    DiagnosticOptions,
    ModelOutputOptions,
    OutputOptions,
    PlotOptions,
    RadiativeFluxOptions,
)
from rubis.rotation_profiles import solid


def set_params():
    """Return the editable configuration used by the RUBIS script."""
    model = CompositePolytropeConfig(
        indices=3.0,
        target_pressures=-np.inf,
    )

    output = OutputOptions(
        diagnostics=DiagnosticOptions(
            virial_test=False,
            gravitational_moments=False,
        ),
        plot=PlotOptions(
            show_harmonics=False,
            show_model=True,
            resolution=501,
            surfaces=True,
        ),
        flux=RadiativeFluxOptions(
            enabled=False,
            plot_lines=True,
            origin=0.05,
            n_lines=15,
            show_effective_temperature=True,
            resolution=(200, 100),
        ),
        model=ModelOutputOptions(
            save=False,
            dimensional=False,
        ),
    )

    return DeformationConfig(
        model=model,
        rotation=RotationConfig(
            profile=solid,
            target=0.9,
            central_diff_rate=1.0,
            scale=1.0,
        ),
        solver=SolverOptions(
            method="auto",
            max_degree=101,
            angular_resolution=101,
            full_rate=3,
            mapping_precision=1.0e-10,
            spline_order=5,
            lagrange_order=3,
            external_domain_res=201,
            rescale_ab=True,
            max_iterations=200,
        ),
        output=output,
    )


if __name__ == "__main__":
    deform(set_params())