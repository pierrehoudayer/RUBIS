"""Deform a uniformly rotating single polytrope."""

from rubis.api import deform
from rubis.config import (
    DeformationConfig,
    PolytropeConfig,
    RotationConfig,
    SolverOptions,
)
from rubis.rotation_profiles import solid


def main():
    config = DeformationConfig(
        model=PolytropeConfig(
            index=3.0,
            radius=1.0,
            mass=1.0,
            n_points=501,
        ),
        rotation=RotationConfig(
            profile=solid,
            target=0.9,
        ),
        solver=SolverOptions(
            method="auto",
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

    print()
    print(f"Method       : {info.method}")
    print(f"Iterations   : {info.iterations}")
    print(f"Polar radius : {info.polar_radius_history[-1]:.10f}")
    print(f"Final error  : {info.error:.3e}")
    print(f"Elapsed time : {info.elapsed_time:.2f} s")
    print(f"Mass         : {model.mass:.10f}")
    print(f"Radius       : {model.radius:.10f}")


if __name__ == "__main__":
    main()