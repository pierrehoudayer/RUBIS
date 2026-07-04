import numpy as np

from rubis.lagrange import lagrange_matrix_P


def test_lagrange_matrix_reproduces_polynomials():
    x = np.linspace(0.0, 1.0, 9)**2
    matrix = lagrange_matrix_P(x, order=2)

    interpolation = matrix[..., 0]
    derivative = matrix[..., 1]

    # The evaluation points are not returned explicitly.
    # Linear interpolation reconstructs them exactly.
    x_eval = interpolation @ x

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
            degree*x_eval**(degree - 1),
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    # Interior rows use four-point stencils and therefore
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
        3*x_eval[interior]**2,
        rtol=1.0e-11,
        atol=1.0e-12,
    )
    
    
def test_higher_order_lagrange_matrix_reproduces_polynomials():
    x = np.linspace(0.0, 1.0, 13)**2
    matrix = lagrange_matrix_P(x, order=3)

    interpolation = matrix[..., 0]
    derivative = matrix[..., 1]
    x_eval = interpolation @ x

    # All rows contain enough nodes to reproduce cubics.
    for degree in range(4):
        values = x**degree
        expected_derivative = (
            np.zeros_like(x_eval)
            if degree == 0
            else degree*x_eval**(degree - 1)
        )

        np.testing.assert_allclose(
            interpolation @ values,
            x_eval**degree,
            rtol=1.0e-11,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            derivative @ values,
            expected_derivative,
            rtol=1.0e-10,
            atol=1.0e-11,
        )

    # Boundary stencils contain fewer nodes. The rows on which
    # the full stencil is available reproduce higher degrees.
    quartic = x**4

    np.testing.assert_allclose(
        (interpolation @ quartic)[1:-1],
        x_eval[1:-1]**4,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        derivative @ quartic,
        4*x_eval**3,
        rtol=1.0e-9,
        atol=1.0e-10,
    )

    quintic = x**5

    np.testing.assert_allclose(
        (interpolation @ quintic)[2:-2],
        x_eval[2:-2]**5,
        rtol=1.0e-9,
        atol=1.0e-11,
    )
    np.testing.assert_allclose(
        (derivative @ quintic)[1:-1],
        5*x_eval[1:-1]**4,
        rtol=1.0e-8,
        atol=1.0e-10,
    )