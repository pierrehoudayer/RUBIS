"""Legendre projection, evaluation, and coupling utilities."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import eval_legendre, roots_legendre


__all__ = [
    "find_r_eq",
    "find_r_pol",
    "legendre_coupling",
    "pl_eval_2D",
    "pl_project_2D",
]


FloatArray = NDArray[np.float64]


def _validate_degree_count(L: int) -> int:
    """Validate the number of retained Legendre degrees."""
    if not isinstance(L, (int, np.integer)) or isinstance(L, bool):
        raise TypeError("L must be an integer.")
    if L < 1:
        raise ValueError("L must be positive.")

    return int(L)


def _validate_derivative_order(der: int) -> int:
    """Validate a Legendre derivative order."""
    if not isinstance(der, (int, np.integer)) or isinstance(der, bool):
        raise TypeError("der must be an integer.")
    if der not in (0, 1, 2):
        raise ValueError("der must be 0, 1, or 2.")

    return int(der)


def _legendre_tables(
    L: int,
    t: ArrayLike,
    *,
    derivative_order: int = 0,
) -> tuple[np.ndarray, ...]:
    """Evaluate Legendre polynomials and the requested derivatives."""
    L = _validate_degree_count(L)
    derivative_order = _validate_derivative_order(derivative_order)
    t = np.asarray(t)

    pl = np.array([
        eval_legendre(l, t)
        for l in range(L)
    ])
    tables = [pl]

    if derivative_order >= 1:
        shape = (L,) + (1,) * t.ndim
        l = np.arange(L).reshape(shape)

        dpl = l * np.roll(pl, 1, axis=0)
        for degree in range(1, L):
            dpl[degree] += t * dpl[degree - 1]

        tables.append(dpl)

    if derivative_order >= 2:
        lp1 = np.where(l != 0, l + 1, 0)

        d2pl = lp1 * np.roll(dpl, 1, axis=0)
        for degree in range(1, L):
            d2pl[degree] += t * d2pl[degree - 1]

        tables.append(d2pl)

    return tuple(tables)


def find_r_eq(
    map_n: ArrayLike,
    L: int,
):
    """Reconstruct the equatorial radius of the outer mapped surface."""
    map_n = np.asarray(map_n)
    if map_n.ndim != 2:
        raise ValueError("map_n must be a two-dimensional array.")

    surface_l = pl_project_2D(map_n[-1], L)

    return pl_eval_2D(surface_l, 0.0)


def find_r_pol(
    map_n: ArrayLike,
    L: int,
):
    """Reconstruct the polar radius of the outer mapped surface."""
    map_n = np.asarray(map_n)
    if map_n.ndim != 2:
        raise ValueError("map_n must be a two-dimensional array.")

    surface_l = pl_project_2D(map_n[-1], L)

    return pl_eval_2D(surface_l, 1.0)


def pl_project_2D(
    f: ArrayLike,
    L: int,
    even: bool = True,
) -> FloatArray:
    """Project values sampled on a Gauss--Legendre grid.

    The last axis is angular. When ``even`` is true, equatorial
    symmetry is assumed and the odd coefficients are set to zero.
    """
    L = _validate_degree_count(L)
    f = np.asarray(f)

    if f.ndim not in (1, 2):
        raise ValueError("f must be one- or two-dimensional.")

    f_2d = np.atleast_2d(f)
    N, M = f_2d.shape

    t, weights = roots_legendre(M)
    norm = (2*np.arange(L) + 1) / 2

    dtype = np.result_type(f_2d.dtype, np.float64)
    f_l = np.zeros((N, L), dtype=dtype)

    degrees = range(0, L, 2) if even else range(L)
    for l in degrees:
        f_l[:, l] = (
            norm[l]
            * (f_2d @ (weights * eval_legendre(l, t)))
        )

    return f_l[0] if f.ndim == 1 else f_l


def pl_eval_2D(
    f_l: ArrayLike,
    t: ArrayLike,
    der: int = 0,
):
    """Evaluate a Legendre expansion and its angular derivatives.

    ``der`` gives the highest derivative order returned, up to the
    second derivative with respect to ``t = cos(theta)``.
    """
    der = _validate_derivative_order(der)
    f_l = np.asarray(f_l)

    if f_l.ndim not in (1, 2):
        raise ValueError("f_l must be one- or two-dimensional.")

    L = f_l.shape[-1]
    tables = _legendre_tables(
        L,
        t,
        derivative_order=der,
    )
    values = tuple(
        f_l @ table
        for table in tables
    )

    return values[0] if der == 0 else values


def legendre_coupling(
    f: ArrayLike,
    L: int,
    der: Sequence[int] = (0, 0),
) -> FloatArray:
    """Compute the even-degree harmonic couplings of ``f``.

    ``der`` selects the derivative order applied to each of the two
    Legendre factors in the angular integral.
    """
    L = _validate_degree_count(L)
    f = np.asarray(f)

    if f.ndim < 1:
        raise ValueError("f must have at least one dimension.")

    try:
        der = tuple(der)
    except TypeError as exc:
        raise TypeError(
            "der must contain two derivative orders."
        ) from exc

    if len(der) != 2:
        raise ValueError(
            "der must contain two derivative orders."
        )

    der = tuple(
        _validate_derivative_order(order)
        for order in der
    )

    M = f.shape[-1]
    t, weights = roots_legendre(M)

    tables = _legendre_tables(
        L,
        t,
        derivative_order=max(der),
    )
    pl1 = tables[der[0]][::2]
    pl2 = tables[der[1]][::2]

    return np.einsum(
        "...k,lk,mk->...lm",
        weights * np.atleast_2d(f),
        pl1,
        pl2,
        optimize="optimal",
    )