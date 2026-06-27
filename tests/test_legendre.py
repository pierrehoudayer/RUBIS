import numpy as np
from scipy.special import eval_legendre, roots_legendre

from legendre import pl_eval_2D, pl_project_2D


def test_legendre_projection_recovers_even_coefficients():
    angular_resolution = 11
    max_degree = 7

    cos_theta, _ = roots_legendre(angular_resolution)

    expected_coefficients = np.zeros(max_degree)
    expected_coefficients[0] = 1.2
    expected_coefficients[2] = -0.3
    expected_coefficients[4] = 0.08
    expected_coefficients[6] = -0.01

    values = sum(
        coefficient * eval_legendre(degree, cos_theta)
        for degree, coefficient in enumerate(expected_coefficients)
    )

    coefficients = pl_project_2D(
        values,
        max_degree,
    )

    np.testing.assert_allclose(
        coefficients,
        expected_coefficients,
        rtol=1.0e-13,
        atol=1.0e-14,
    )


def test_legendre_reconstruction_on_independent_grid():
    coefficients = np.array([
        1.2,
        0.0,
        -0.3,
        0.0,
        0.08,
        0.0,
        -0.01,
    ])

    cos_theta = np.linspace(-1.0, 1.0, 31)

    reconstructed = pl_eval_2D(
        coefficients,
        cos_theta,
    )
    expected = sum(
        coefficient * eval_legendre(degree, cos_theta)
        for degree, coefficient in enumerate(coefficients)
    )

    np.testing.assert_allclose(
        reconstructed,
        expected,
        rtol=1.0e-14,
        atol=1.0e-14,
    )


def test_legendre_projection_discards_odd_degrees_by_default():
    angular_resolution = 11
    max_degree = 6

    cos_theta, _ = roots_legendre(angular_resolution)

    values = (
        1.0
        + 0.4 * eval_legendre(1, cos_theta)
        - 0.2 * eval_legendre(2, cos_theta)
        + 0.1 * eval_legendre(3, cos_theta)
    )

    coefficients = pl_project_2D(
        values,
        max_degree,
    )

    expected_coefficients = np.array([
        1.0,
        0.0,
        -0.2,
        0.0,
        0.0,
        0.0,
    ])

    np.testing.assert_allclose(
        coefficients,
        expected_coefficients,
        rtol=1.0e-13,
        atol=1.0e-14,
    )