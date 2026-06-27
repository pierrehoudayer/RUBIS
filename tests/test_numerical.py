import numpy as np

from numerical import integrate, interpolate_func, lagrange_matrix_P


def test_lagrange_matrix_reproduces_polynomials():
    x = np.linspace(0.0, 1.0, 9) ** 2

    matrix = lagrange_matrix_P(x, order=2)

    interpolation = matrix[..., 0]
    derivative = matrix[..., 1]

    # The evaluation points are not returned explicitly.
    # Since linear interpolation is exact, they can be
    # reconstructed by applying the interpolation matrix to x.
    x_eval = interpolation @ x

    # Constants
    np.testing.assert_allclose(
        interpolation @ np.ones_like(x),
        np.ones_like(x_eval),
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        derivative @ np.ones_like(x),
        np.zeros_like(x_eval),
        rtol=0.0,
        atol=1.0e-13,
    )

    # Linear and quadratic polynomials are reproduced on every row.
    for degree in (1, 2):
        values = x**degree

        np.testing.assert_allclose(
            interpolation @ values,
            x_eval**degree,
            rtol=1.0e-12,
            atol=1.0e-13,
        )
        np.testing.assert_allclose(
            derivative @ values,
            degree * x_eval**(degree - 1),
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    # Interior rows use four-point stencils and therefore also
    # reproduce cubic polynomials exactly.
    cubic = x**3
    interior = slice(1, -1)

    np.testing.assert_allclose(
        (interpolation @ cubic)[interior],
        x_eval[interior]**3,
        rtol=1.0e-12,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        (derivative @ cubic)[interior],
        3.0 * x_eval[interior]**2,
        rtol=1.0e-11,
        atol=1.0e-12,
    )
    
    
def test_integrate_reproduces_cubic_polynomial():
    x = np.linspace(0.0, 1.0, 21) ** 2

    def function(x):
        return 3.0 * x**2 - 2.0 * x + 1.0

    def primitive(x):
        return x**3 - x**2 + x

    y = function(x)

    full_integral = integrate(x, y)
    expected_full_integral = primitive(1.0) - primitive(0.0)

    np.testing.assert_allclose(
        full_integral,
        expected_full_integral,
        rtol=1.0e-13,
        atol=1.0e-14,
    )

    lower_bound = 0.13
    upper_bound = 0.82

    partial_integral = integrate(
        x,
        y,
        a=lower_bound,
        b=upper_bound,
    )
    expected_partial_integral = (
        primitive(upper_bound) - primitive(lower_bound)
    )

    np.testing.assert_allclose(
        partial_integral,
        expected_partial_integral,
        rtol=1.0e-13,
        atol=1.0e-14,
    )
    
    
def test_interpolate_func_reproduces_cubic_and_derivative():
    x = np.linspace(0.0, 1.0, 17) ** 2

    def function(x):
        return 1.0 - 2.0 * x + 3.0 * x**2 - 0.5 * x**3

    def derivative(x):
        return -2.0 + 6.0 * x - 1.5 * x**2

    interpolant = interpolate_func(
        x,
        function(x),
        der=(0, 1),
        k=3,
    )

    x_eval = np.linspace(0.0, 1.0, 41)
    values, derivatives = interpolant(x_eval)

    np.testing.assert_allclose(
        values,
        function(x_eval),
        rtol=1.0e-12,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        derivatives,
        derivative(x_eval),
        rtol=1.0e-11,
        atol=1.0e-12,
    )