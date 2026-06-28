from pathlib import Path

import numpy as np

from rubis._utils import DotDict
from model_deform_radial import radial_method
from model_deform_spheroidal import spheroidal_method
from rubis.rotation_profiles import solid


RADIAL_REFERENCE_PATH = (
    Path(__file__).parent
    / "tests"
    / "reference"
    / "radial_n1_solid_omega_0p3_legacy.npz"
)

SPHEROIDAL_REFERENCE_PATH = (
    Path(__file__).parent
    / "tests"
    / "reference"
    / "spheroidal_composite_solid_omega_0p3_legacy.npz"
)


def generate_radial_reference():
    model = DotDict(
        indices=1.0,
        target_pressures=-np.inf,
        density_jumps=None,
        radius=1.0,
        mass=1.0,
        resolution=65,
    )

    output = DotDict(
        show_harmonics=False,
        virial_test=False,
        show_model=False,
        gravitational_moments=False,
        save_model=False,
    )

    result = radial_method(
        model,
        solid,
        0.3,
        0.0,
        1.0,
        9,
        9,
        1,
        1.0e-10,
        3,
        2,
        output,
        21,
        True,
    )

    RADIAL_REFERENCE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        RADIAL_REFERENCE_PATH,
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

    print(
        "Radial reference written to "
        f"{RADIAL_REFERENCE_PATH}"
    )
    
    
def generate_spheroidal_reference():
    model = DotDict(
        indices=(1.0, 1.0),
        target_pressures=(-1.0, -np.inf),
        density_jumps=(0.4,),
        radius=1.0,
        mass=1.0,
        resolution=65,
    )

    output = DotDict(
        show_harmonics=False,
        virial_test=False,
        show_model=False,
        gravitational_moments=False,
        save_model=False,
    )

    result = spheroidal_method(
        model,
        solid,
        0.3,
        0.0,
        1.0,
        9,
        9,
        1,
        1.0e-10,
        3,
        2,
        output,
        21,
        True,
    )

    SPHEROIDAL_REFERENCE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        SPHEROIDAL_REFERENCE_PATH,
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

    print(
        "Spheroidal reference written to "
        f"{SPHEROIDAL_REFERENCE_PATH}"
    )


if __name__ == "__main__":
    generate_radial_reference()
    generate_spheroidal_reference()