import numpy as np
import pytest

from rubis.config import RotationConfig
from rubis.rotation import initialize_rotation_state
from rubis.rotation_profiles import (
    lorentzian,
    plateau,
    solid,
)


@pytest.mark.parametrize(
    ("profile", "profile_args"),
    [
        (solid, ()),
        (lorentzian, (0.4,)),
        (plateau, (0.4, 0.25)),
    ],
)
def test_rotation_state_binds_profile_parameters(
    profile,
    profile_args,
):
    r = np.linspace(0.0, 1.0, 11)
    t = 0.37
    omega_eq = 0.63

    rot = initialize_rotation_state(
        RotationConfig(
            profile=profile,
            central_diff_rate=0.4,
            scale=0.25,
        ),
        omega_eq=omega_eq,
    )

    phi_c, dphi_c = rot.phi_c(r, t)
    omega = rot.omega(r, t)

    expected_phi_c, expected_dphi_c = profile(
        r,
        t,
        omega_eq,
        *profile_args,
    )
    expected_omega = profile(
        r,
        t,
        omega_eq,
        *profile_args,
        return_profile=True,
    )

    np.testing.assert_array_equal(
        phi_c,
        expected_phi_c,
    )
    np.testing.assert_array_equal(
        dphi_c,
        expected_dphi_c,
    )
    np.testing.assert_array_equal(
        omega,
        expected_omega,
    )
    
    
def test_rotation_state_evaluates_mapping_fields():
    r = np.linspace(0.0, 1.0, 5)
    t = np.array([-0.5, 0.0, 0.5])
    r2d = np.repeat(r[:, None], t.size, axis=1)

    rot = initialize_rotation_state(
        RotationConfig(profile=solid),
        omega_eq=0.3,
    )

    phi_c2d = rot.phi_c2d(r2d, t)
    omega2d = rot.omega2d(r2d, t)

    assert phi_c2d.shape == r2d.shape
    assert omega2d.shape == r2d.shape

    for j, t_j in enumerate(t):
        expected_phi_c = rot.phi_c(
            r2d[:, j],
            t_j,
        )[0]
        expected_omega = rot.omega(
            r2d[:, j],
            t_j,
        )

        np.testing.assert_allclose(
            phi_c2d[:, j],
            expected_phi_c,
        )
        np.testing.assert_allclose(
            omega2d[:, j],
            expected_omega,
        )