"""Spline and axisymmetric quadrature utilities."""

import numpy as np
from scipy.interpolate import splint, splrep
from scipy.special import roots_legendre


__all__ = [
    "integrate",
    "integrate_axisymmetric",
]


def integrate(
    x,
    y,
    a=None,
    b=None,
    k: int = 3,
):
    """Integrate sampled values with a B-spline representation."""
    tck = splrep(x, y, k=k)

    if a is None:
        a = x[0]
    if b is None:
        b = x[-1]

    return splint(a, b, tck)


def integrate_axisymmetric(
    r2d,
    y,
    domains=None,
):
    """Integrate a field over an axisymmetric mapped volume."""
    r2d = np.asarray(r2d)
    n_points, angular_resolution = r2d.shape
    t, weights = roots_legendre(angular_resolution)

    if domains is None:
        domains = (np.arange(n_points),)

    if callable(y):
        radial_integral = np.array([
            sum(
                integrate(
                    r_j[domain],
                    y(r_j, t_j, domain) * r_j[domain]**2,
                    k=5,
                )
                for domain in domains
            )
            for r_j, t_j in zip(r2d.T, t)
        ])
    else:
        y = np.asarray(y)

        if y.ndim < 2:
            y = np.tile(
                y,
                (angular_resolution, 1),
            ).T

        radial_integral = np.array([
            sum(
                integrate(
                    r_j[domain],
                    y_j[domain] * r_j[domain]**2,
                    k=5,
                )
                for domain in domains
            )
            for r_j, y_j in zip(r2d.T, y.T)
        ])

    return 2*np.pi * radial_integral @ weights