import numpy as np

from rotation_profiles import solid


def test_solid_rotation_profile_is_constant():
    r = np.linspace(0.0, 1.0, 11)
    cos_theta = 0.37
    omega = 0.63

    profile = solid(
        r,
        cos_theta,
        omega,
        return_profile=True,
    )

    np.testing.assert_allclose(
        profile,
        omega,
        rtol=0.0,
        atol=0.0,
    )


def test_solid_rotation_potential():
    r = np.linspace(0.0, 1.0, 11)
    cos_theta = 0.37
    omega = 0.63

    potential, derivative = solid(r, cos_theta, omega)

    sin_theta_squared = 1.0 - cos_theta**2

    expected_potential = (
        -0.5 * omega**2 * r**2 * sin_theta_squared
    )
    expected_derivative = (
        -omega**2 * r * sin_theta_squared
    )

    np.testing.assert_allclose(
        potential,
        expected_potential,
        rtol=1.0e-14,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        derivative,
        expected_derivative,
        rtol=1.0e-14,
        atol=1.0e-15,
    )


def test_solid_rotation_derivative_matches_finite_difference():
    r = np.array([0.1, 0.4, 0.9])
    cos_theta = 0.37
    omega = 0.63
    step = 1.0e-6

    potential_plus, _ = solid(
        r + step,
        cos_theta,
        omega,
    )
    potential_minus, _ = solid(
        r - step,
        cos_theta,
        omega,
    )
    _, derivative = solid(r, cos_theta, omega)

    numerical_derivative = (
        potential_plus - potential_minus
    ) / (2.0 * step)

    np.testing.assert_allclose(
        derivative,
        numerical_derivative,
        rtol=1.0e-9,
        atol=1.0e-11,
    )