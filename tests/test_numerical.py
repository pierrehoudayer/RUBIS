import numpy as np
from scipy.special import roots_legendre

from rubis.numerical import (
    del_u_over_v,
    expI,
    expinv,
    integrate,
    integrate2D,
    interpolate_func,
    lagrange_matrix_P,
    lnxn,
)


def test_lnxn_matches_definition_and_continuation():
    x = np.array([0.0, 0.2, 0.7, 1.0])

    values = lnxn(x, n=2, a=1.5)
    expected = np.zeros_like(x)
    expected[1:] = x[1:]**1.5 * np.log(x[1:])**2

    np.testing.assert_allclose(
        values,
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_expinv_and_expI_are_consistent():
    x = np.linspace(0.05, 2.0, 41)
    k = 2
    a = 1.7

    expected = np.exp(-a*x**(-1/k))

    np.testing.assert_allclose(
        expinv(x, k=k, a=a),
        expected,
        rtol=0.0,
        atol=0.0,
    )

    step = 3.0e-4

    def primitive(x):
        return x * expI(x, k=k, a=a)

    derivative = (
        -primitive(x + 2*step)
        + 8*primitive(x + step)
        - 8*primitive(x - step)
        + primitive(x - 2*step)
    ) / (12*step)

    np.testing.assert_allclose(
        derivative,
        expected,
        rtol=2.0e-9,
        atol=2.0e-12,
    )


def test_del_u_over_v_reproduces_first_two_derivatives():
    x = np.linspace(0.1, 1.0, 11)

    du = (
        x**2,
        2*x,
        2*np.ones_like(x),
    )
    dv = (
        x + 1,
        np.ones_like(x),
        np.zeros_like(x),
    )

    expected = (
        x**2 / (x + 1),
        1 - 1/(x + 1)**2,
        2/(x + 1)**3,
    )

    for der in range(3):
        np.testing.assert_allclose(
            del_u_over_v(du, dv, der),
            expected[der],
            rtol=1.0e-14,
            atol=1.0e-14,
        )


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


def test_integrate_reproduces_cubic_polynomial():
    x = np.linspace(0.0, 1.0, 21)**2

    def function(x):
        return 3*x**2 - 2*x + 1

    def primitive(x):
        return x**3 - x**2 + x

    y = function(x)

    np.testing.assert_allclose(
        integrate(x, y),
        primitive(1.0) - primitive(0.0),
        rtol=1.0e-13,
        atol=1.0e-14,
    )

    lower_bound = 0.13
    upper_bound = 0.82

    np.testing.assert_allclose(
        integrate(
            x,
            y,
            a=lower_bound,
            b=upper_bound,
        ),
        primitive(upper_bound) - primitive(lower_bound),
        rtol=1.0e-13,
        atol=1.0e-14,
    )


def test_integrate2D_reproduces_axisymmetric_polynomial():
    r = np.linspace(0.0, 1.0, 21)
    cos_theta, _ = roots_legendre(9)

    r2d = np.broadcast_to(
        r[:, None],
        (r.size, cos_theta.size),
    )
    y = r2d * (1 + cos_theta**2)

    np.testing.assert_allclose(
        integrate2D(r2d, y),
        4*np.pi/3,
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_integrate2D_handles_discontinuous_domains():
    lower = np.linspace(0.0, 0.5, 7)
    upper = np.linspace(0.5, 1.0, 7)
    r = np.hstack((lower, upper))

    r2d = np.broadcast_to(
        r[:, None],
        (r.size, 7),
    )
    domains = (
        np.arange(lower.size),
        np.arange(lower.size, r.size),
    )

    def piecewise_constant(r, cos_theta, domain):
        value = 1.0 if domain[0] == 0 else 2.0

        return value * np.ones(domain.size)

    integral = integrate2D(
        r2d,
        piecewise_constant,
        domains=domains,
    )

    np.testing.assert_allclose(
        integral,
        2.5*np.pi,
        rtol=1.0e-13,
        atol=1.0e-13,
    )


def test_interpolate_func_reproduces_cubic_and_derivative():
    x = np.linspace(0.0, 1.0, 17)**2

    def function(x):
        return 1 - 2*x + 3*x**2 - 0.5*x**3

    def derivative(x):
        return -2 + 6*x - 1.5*x**2

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


def test_interpolate_func_applies_primitive_condition():
    x = np.linspace(0.0, 1.0, 17)**2

    def function(x):
        return 1 - 2*x + 3*x**2

    def primitive(x):
        return x - x**2 + x**3

    index = 6
    value = 2.3

    antiderivative = interpolate_func(
        x,
        function(x),
        der=-1,
        k=3,
        prim_cond=(index, value),
    )

    x_eval = np.linspace(0.0, 1.0, 41)
    expected = (
        primitive(x_eval)
        - primitive(x[index])
        + value
    )

    np.testing.assert_allclose(
        antiderivative(x_eval),
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_interpolate_func_preserves_empty_evaluations():
    x = np.linspace(0.0, 1.0, 7)
    y = x**2
    empty = np.array([])

    value = interpolate_func(
        x,
        y,
        der=0,
    )(empty)
    values = interpolate_func(
        x,
        y,
        der=(0, 1),
    )(empty)

    assert value.shape == (0,)
    assert values.shape == (2, 0)