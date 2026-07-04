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
    ("profile", "profile_parameters"),
    [
        (solid, {}),
        (
            lorentzian,
            {
                "alpha": 0.4,
            },
        ),
        (
            plateau,
            {
                "alpha": 0.4,
                "scale": 0.25,
            },
        ),
    ],
)
def test_rotation_state_binds_profile_parameters(
    profile,
    profile_parameters,
):
    r = np.linspace(0.0, 1.0, 11)
    t = 0.37
    omega_eq = 0.63

    rot = initialize_rotation_state(
        RotationConfig(
            profile=profile,
            profile_parameters=profile_parameters,
        ),
        omega_eq=omega_eq,
    )

    phi_c, phi_c_r = rot.phi_c(r, t)
    omega = rot.omega(r, t)

    expected_phi_c, expected_phi_c_r = profile(
        r,
        t,
        omega_eq,
        **profile_parameters,
    )
    expected_omega = profile(
        r,
        t,
        omega_eq,
        **profile_parameters,
        return_profile=True,
    )

    np.testing.assert_array_equal(
        phi_c,
        expected_phi_c,
    )
    np.testing.assert_array_equal(
        phi_c_r,
        expected_phi_c_r,
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
        
        
def test_rotation_state_evaluates_two_dimensional_mapping():
    r = np.linspace(0.0, 1.0, 7)
    t = np.linspace(-1.0, 1.0, 9)
    r2d = r[:, None] * (
        1.0 - 0.1*(1 - t**2)
    )

    rot = initialize_rotation_state(
        RotationConfig(
            profile=lorentzian,
            profile_parameters={
                "alpha": 0.4,
            },
        ),
        omega_eq=0.63,
    )

    phi_c, phi_c_r = rot.phi_c2d_with_derivative(r2d, t)

    expected = [
        rot.phi_c(r2d[:, j], t[j])
        for j in range(t.size)
    ]
    expected_phi_c, expected_phi_c_r = np.moveaxis(
        np.asarray(expected),
        (0, 1, 2),
        (2, 0, 1),
    )

    np.testing.assert_allclose(
        phi_c,
        expected_phi_c,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        phi_c_r,
        expected_phi_c_r,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        rot.phi_c2d(r2d, t),
        phi_c,
    )