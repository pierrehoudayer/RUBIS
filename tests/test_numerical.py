import numpy as np

from numerical import lagrange_matrix_P


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