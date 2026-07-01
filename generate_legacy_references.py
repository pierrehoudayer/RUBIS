"""Regenerate solver reference files explicitly.

The existing ``*_legacy.npz`` files are historical regression references.
Do not overwrite them unless the numerical baseline is intentionally changed.
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from rubis.api import deform
from rubis.config import (
    CompositePolytropeConfig,
    DeformationConfig,
    PolytropeConfig,
    OutputOptions,
    RotationConfig,
    SolverOptions,
)
from rubis.solvers import solve_radial, solve_spheroidal
from rubis.results import RadialResult, SpheroidalResult
from rubis.rotation_profiles import solid


REFERENCE_DIR = Path(__file__).parent / "tests" / "reference"

RADIAL_REFERENCE_PATH = (
    REFERENCE_DIR
    / "radial_n1_solid_omega_0p3_legacy.npz"
)

SPHEROIDAL_REFERENCE_PATH = (
    REFERENCE_DIR
    / "spheroidal_composite_solid_omega_0p3_legacy.npz"
)


def _check_output_path(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists. Pass --overwrite to replace "
            "the historical reference intentionally."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def _save_radial_reference(
    path: Path,
    result: RadialResult,
    overwrite: bool,
) -> None:
    _check_output_path(path, overwrite)

    np.savez_compressed(
        path,
        zeta=result.zeta,
        cos_theta=result.cos_theta,
        mapping=result.mapping,
        density=result.density,
        pressure=result.pressure,
        effective_potential=result.effective_potential,
        gravitational_potential_harmonics=(
            result.gravitational_potential_harmonics
        ),
        gravitational_potential_derivative_harmonics=(
            result.gravitational_potential_derivative_harmonics
        ),
        polar_radius=result.polar_radius_history[-1],
        mass=result.mass,
        radius=result.radius,
        rotation_rate=result.rotation_rate,
    )

    print(f"Radial reference written to {path}")


def _save_spheroidal_reference(
    path: Path,
    result: SpheroidalResult,
    overwrite: bool,
) -> None:
    _check_output_path(path, overwrite)

    np.savez_compressed(
        path,
        zeta=result.zeta,
        internal_zeta=result.internal_zeta,
        external_zeta=result.external_zeta,
        cos_theta=result.cos_theta,
        mapping=result.mapping,
        full_mapping=result.full_mapping,
        density=result.density,
        pressure=result.pressure,
        effective_potential=result.effective_potential,
        effective_potential_derivative=(
            result.effective_potential_derivative
        ),
        gravitational_potential_harmonics=(
            result.gravitational_potential_harmonics
        ),
        gravitational_potential_derivative_harmonics=(
            result.gravitational_potential_derivative_harmonics
        ),
        internal_mask=result.internal_mask,
        external_mask=result.external_mask,
        polar_radius=result.polar_radius_history[-1],
        mass=result.mass,
        radius=result.radius,
        rotation_rate=result.rotation_rate,
    )

    print(f"Spheroidal reference written to {path}")


def generate_radial_reference(
    *,
    overwrite: bool = False,
) -> None:
    model = PolytropeConfig(
        index=1.0,
        radius=1.0,
        mass=1.0,
        n_points=65,
    )

    result = deform(
        DeformationConfig(
            model=model,
            rotation=RotationConfig(
                profile=solid,
                target=0.3,
            ),
            solver=SolverOptions(
                method="radial",
                max_degree=9,
                angular_resolution=9,
                full_rate=1,
                mapping_precision=1.0e-10,
                spline_order=3,
                lagrange_order=2,
                external_domain_res=21,
                rescale_ab=True,
            ),
        )
    )

    _save_radial_reference(
        RADIAL_REFERENCE_PATH,
        result,
        overwrite,
    )


def generate_spheroidal_reference(
    *,
    overwrite: bool = False,
) -> None:
    model = CompositePolytropeConfig(
        indices=(1.0, 1.0),
        target_pressures=(-1.0, -np.inf),
        density_jumps=(0.4,),
        radius=1.0,
        mass=1.0,
        n_points=65,
    )

    result = deform(
        DeformationConfig(
            model=model,
            rotation=RotationConfig(
                profile=solid,
                target=0.3,
            ),
            solver=SolverOptions(
                method="spheroidal",
                max_degree=9,
                angular_resolution=9,
                full_rate=1,
                mapping_precision=1.0e-10,
                spline_order=3,
                lagrange_order=2,
                external_domain_res=21,
                rescale_ab=True,
            ),
        )
    )

    _save_spheroidal_reference(
        SPHEROIDAL_REFERENCE_PATH,
        result,
        overwrite,
    )


def main() -> None:
    parser = ArgumentParser(
        description=(
            "Regenerate the numerical regression references. "
            "Existing files are protected by default."
        )
    )
    parser.add_argument(
        "--case",
        choices=("all", "radial", "spheroidal"),
        default="all",
        help="Reference case to generate.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing reference files.",
    )

    args = parser.parse_args()

    if args.case in ("all", "radial"):
        generate_radial_reference(
            overwrite=args.overwrite,
        )

    if args.case in ("all", "spheroidal"):
        generate_spheroidal_reference(
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()