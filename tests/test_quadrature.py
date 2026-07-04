import numpy as np
from scipy.special import roots_legendre

from rubis.quadrature import integrate, integrate_axisymmetric


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
    
    
def test_integrate_axisymmetric_reproduces_axisymmetric_polynomial():
    r = np.linspace(0.0, 1.0, 21)
    cos_theta, _ = roots_legendre(9)

    r2d = np.broadcast_to(
        r[:, None],
        (r.size, cos_theta.size),
    )
    y = r2d * (1 + cos_theta**2)

    np.testing.assert_allclose(
        integrate_axisymmetric(r2d, y),
        4*np.pi/3,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    
    
def test_integrate_axisymmetric_handles_discontinuous_domains():
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

    integral = integrate_axisymmetric(
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