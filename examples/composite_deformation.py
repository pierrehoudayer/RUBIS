"""Deform a composite model with the spheroidal solver."""

import numpy as np

from rubis.api import deform
from rubis.config import (
    CompositePolytropeConfig,
    DeformationConfig,
    RotationConfig,
    SolverOptions,
)
from rubis.rotation_profiles import solid


def main():
    config = DeformationConfig(
        model=CompositePolytropeConfig(
            indices=(1.0, 1.0),
            target_pressures=(
                -1.0,
                -np.inf,
            ),
            density_jumps=(0.4,),
            radius=1.0,
            mass=1.0,
            n_points=501,
        ),
        rotation=RotationConfig(
            profile=solid,
            target=0.3,
        ),
        solver=SolverOptions(
            method="auto",
            max_degree=51,
            angular_resolution=51,
            full_rate=1,
            mapping_precision=1.0e-10,
            spline_order=5,
            lagrange_order=2,
            external_domain_res=101,
            verbose=True,
        ),
    )

    model, vacuum, info = deform(config)

    assert vacuum is not None

    print()
    print(f"Method          : {info.method}")
    print(f"Material domains: {model.n_domains}")
    print(f"Material points : {model.n_points}")
    print(f"Vacuum points   : {vacuum.n_points}")
    print(f"Polar radius    : {info.polar_radius_history[-1]:.10f}")
    print(f"Final error     : {info.error:.3e}")


if __name__ == "__main__":
    main()