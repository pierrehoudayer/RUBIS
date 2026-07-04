import numpy as np

from rubis.special import del_u_over_v, expI, expinv, lnxn


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