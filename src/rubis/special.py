"""Small analytical functions used by RUBIS numerical schemes."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import expn


__all__ = [
    "del_u_over_v",
    "expI",
    "expinv",
    "lnxn",
]


def lnxn(
    x: ArrayLike,
    n: int = 1,
    a: float = 1.0,
):
    r"""Evaluate \(x^a \log^n(x)\), continuously extended by zero at \(x=0\)."""
    x = np.asarray(x)

    with np.errstate(all="ignore"):
        return np.where(
            x == 0,
            0.0,
            np.log(x)**n * x**a,
        )


def expinv(
    x: ArrayLike,
    k: int = 1,
    a: float = 1.0,
):
    r"""Evaluate \(\exp[-a x^{-1/k}]\)."""
    x = np.asarray(x)

    with np.errstate(all="ignore"):
        u = x**-(1/k)

    return np.exp(-a*u)


def expI(
    x: ArrayLike,
    k: int = 1,
    a: float = 1.0,
):
    """Evaluate the normalized primitive kernel associated with ``expinv``."""
    x = np.asarray(x)

    with np.errstate(all="ignore"):
        u = x**-(1/k)

    return k * expn(k + 1, a*u)


def del_u_over_v(
    u_derivatives: Sequence,
    v_derivatives: Sequence,
    der: int,
):
    """Evaluate a derivative of a quotient from derivatives of its factors."""
    if der not in (0, 1, 2):
        raise ValueError("der must be 0, 1, or 2.")

    u = u_derivatives
    v = v_derivatives

    if der == 0:
        return u[0] / v[0]

    if der == 1:
        return (
            + u[1] / v[0]
            - u[0]*v[1] / v[0]**2
        )

    return (
        + u[2] / v[0]
        - (2*u[1]*v[1] + u[0]*v[2]) / v[0]**2
        + 2*u[0]*v[1]**2 / v[0]**3
    )