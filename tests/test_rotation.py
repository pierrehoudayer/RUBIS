import numpy as np

from rubis.config import RotationConfig
from rubis.rotation import initialize_rotation_state
from rubis.rotation_profiles import solid


def test_rotation_state_updates_equatorial_rate_immutably():
    state = initialize_rotation_state(
        RotationConfig(
            profile=solid,
        )
    )

    updated = state.with_omega_eq(0.3)

    assert state.omega_eq == 0.0
    assert updated.omega_eq == 0.3


def test_solid_rotation_state_evaluates_uniform_omega():
    r = np.linspace(0.0, 1.0, 5)

    state = initialize_rotation_state(
        RotationConfig(
            profile=solid,
        ),
        omega_eq=0.3,
    )

    np.testing.assert_allclose(
        state.omega(r, 0.0),
        0.3,
    )