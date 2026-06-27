from pathlib import Path

import numpy as np

from helpers import DotDict
from model_deform_radial import radial_method
from rotation_profiles import solid


REFERENCE_PATH = (
    Path(__file__).parent
    / "tests"
    / "reference"
    / "radial_n1_solid_omega_0p3_legacy.npz"
)


def main():
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

    REFERENCE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        REFERENCE_PATH,
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

    print(f"Reference written to {REFERENCE_PATH}")


if __name__ == "__main__":
    main()