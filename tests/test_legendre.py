import numpy as np

from numpy.polynomial.legendre import Legendre
from scipy.special import eval_legendre, roots_legendre

from rubis.legendre import (
    legendre_coupling,
    find_r_eq,
    find_r_pol,
    pl_eval_2D,
    pl_project_2D,
)


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


def test_legendre_projection_recovers_odd_coefficients_when_requested():
    angular_resolution = 12
    max_degree = 7
    cos_theta, _ = roots_legendre(angular_resolution)

    expected_coefficients = np.array([
        1.2,
        -0.3,
        0.08,
        0.04,
        -0.01,
        0.02,
        -0.005,
    ])

    values = sum(
        coefficient * eval_legendre(degree, cos_theta)
        for degree, coefficient in enumerate(expected_coefficients)
    )

    coefficients = pl_project_2D(
        values,
        max_degree,
        even=False,
    )

    np.testing.assert_allclose(
        coefficients,
        expected_coefficients,
        rtol=1.0e-13,
        atol=5.0e-14,
    )


def test_legendre_projection_supports_radial_batches():
    angular_resolution = 11
    max_degree = 7
    cos_theta, _ = roots_legendre(angular_resolution)

    expected_coefficients = np.array([
        [1.2, 0.0, -0.3, 0.0, 0.08, 0.0, -0.01],
        [0.7, 0.0, 0.15, 0.0, -0.04, 0.0, 0.02],
    ])

    values = np.array([
        sum(
            coefficient * eval_legendre(degree, cos_theta)
            for degree, coefficient in enumerate(coefficients)
        )
        for coefficients in expected_coefficients
    ])

    coefficients = pl_project_2D(
        values,
        max_degree,
    )

    assert coefficients.shape == expected_coefficients.shape
    np.testing.assert_allclose(
        coefficients,
        expected_coefficients,
        rtol=1.0e-13,
        atol=1.0e-14,
    )


def test_legendre_evaluation_returns_first_and_second_derivatives():
    coefficients = np.array([
        0.4,
        -0.3,
        0.2,
        0.1,
        -0.05,
    ])
    cos_theta = np.linspace(-0.8, 0.8, 17)
    polynomial = Legendre(coefficients)

    values, first = pl_eval_2D(
        coefficients,
        cos_theta,
        der=1,
    )
    values_2, first_2, second = pl_eval_2D(
        coefficients,
        cos_theta,
        der=2,
    )

    np.testing.assert_allclose(
        values,
        polynomial(cos_theta),
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        first,
        polynomial.deriv(1)(cos_theta),
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        values_2,
        values,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        first_2,
        first,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        second,
        polynomial.deriv(2)(cos_theta),
        rtol=1.0e-14,
        atol=1.0e-14,
    )


def test_legendre_coupling_recovers_even_orthogonality():
    angular_resolution = 15
    max_degree = 8
    values = np.ones(angular_resolution)
    even_degrees = np.arange(0, max_degree, 2)
    expected = np.diag(2.0 / (2*even_degrees + 1))

    coupling = legendre_coupling(
        values,
        max_degree,
    )

    assert coupling.shape == (
        1,
        even_degrees.size,
        even_degrees.size,
    )
    np.testing.assert_allclose(
        coupling[0],
        expected,
        rtol=1.0e-13,
        atol=1.0e-14,
    )


def test_legendre_coupling_supports_derivative_orders():
    angular_resolution = 15
    max_degree = 8
    cos_theta, weights = roots_legendre(angular_resolution)
    values = 1.0 + 0.2*cos_theta - 0.1*cos_theta**2
    even_degrees = np.arange(0, max_degree, 2)

    left = np.array([
        Legendre.basis(degree).deriv(1)(cos_theta)
        for degree in even_degrees
    ])
    right = np.array([
        Legendre.basis(degree).deriv(2)(cos_theta)
        for degree in even_degrees
    ])
    expected = np.einsum(
        "k,lk,mk->lm",
        weights * values,
        left,
        right,
    )

    coupling = legendre_coupling(
        values,
        max_degree,
        der=(1, 2),
    )

    np.testing.assert_allclose(
        coupling[0],
        expected,
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_equatorial_and_polar_radii_follow_surface_expansion():
    angular_resolution = 15
    max_degree = 7
    cos_theta, _ = roots_legendre(angular_resolution)

    coefficients = np.array([
        1.0,
        0.0,
        -0.12,
        0.0,
        0.03,
        0.0,
        -0.005,
    ])
    surface = sum(
        coefficient * eval_legendre(degree, cos_theta)
        for degree, coefficient in enumerate(coefficients)
    )
    mapping = np.vstack((
        0.5 * surface,
        surface,
    ))
    polynomial = Legendre(coefficients)

    np.testing.assert_allclose(
        find_r_eq(mapping, max_degree),
        polynomial(0.0),
        rtol=1.0e-13,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        find_r_pol(mapping, max_degree),
        polynomial(1.0),
        rtol=1.0e-13,
        atol=1.0e-14,
    )