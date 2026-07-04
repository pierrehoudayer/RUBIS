"""Cylindrical rotation laws and centrifugal potentials."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from .interpolation import interpolate_func
from .special import expI, expinv


__all__ = [
    "lorentzian",
    "plateau",
    "solid",
    "tabulated",
]


def _cylindrical_geometry(
    r: ArrayLike,
    t: ArrayLike,
):
    """Return the squared cylindrical radius and its radial derivative."""
    s2   = r**2 * (1 - t**2)
    s2_r = 2*r  * (1 - t**2)

    return s2, s2_r


def solid(
    r: ArrayLike,
    t: ArrayLike,
    omega_eq: float,
    return_profile: bool = False,
):
    """Evaluate solid rotation and its centrifugal potential."""
    s2, s2_r = _cylindrical_geometry(r, t)

    if return_profile:
        return omega_eq * np.ones_like(r)

    phi_c   = -0.5 * s2   * omega_eq**2
    phi_c_r = -0.5 * s2_r * omega_eq**2

    return phi_c, phi_c_r


def lorentzian(
    r: ArrayLike,
    t: ArrayLike,
    omega_eq: float,
    alpha: float,
    return_profile: bool = False,
):
    """Evaluate a Lorentzian differential-rotation law.

    ``alpha`` is the relative difference between the central and
    equatorial angular velocities.
    """
    s2, s2_r = _cylindrical_geometry(r, t)
    norm = (1 + alpha)**2

    if return_profile:
        return omega_eq * (1 + alpha) / (1 + alpha*s2)

    phi_c   = -0.5 * s2   * norm / (1 + alpha*s2)    * omega_eq**2
    phi_c_r = -0.5 * s2_r * norm / (1 + alpha*s2)**2 * omega_eq**2

    return phi_c, phi_c_r


def plateau(
    r: ArrayLike,
    t: ArrayLike,
    omega_eq: float,
    alpha: float,
    scale: float,
    return_profile: bool = False,
    k: int = 1,
):
    """Evaluate a differential-rotation law with a central plateau.

    ``alpha`` fixes the central-to-equatorial contrast, while ``scale``
    controls the cylindrical extent of the plateau.
    """
    corr = np.exp(scale**(2/k))
    omega_0 = (1 + alpha) * omega_eq
    delta_omega = alpha * omega_eq * corr

    s2, s2_r = _cylindrical_geometry(r, t)
    x = s2 / scale**2

    if return_profile:
        return omega_0 - delta_omega * expinv(x, k)

    I1  = expI(x, k, 1)
    I2  = expI(x, k, 2)
    II1 = expinv(x, k, 1)
    II2 = expinv(x, k, 2)

    phi_c   = -0.5 * s2   * (
        omega_0**2 - 2*omega_0*delta_omega*I1  + delta_omega**2*I2
    )
    phi_c_r = -0.5 * s2_r * (
        omega_0**2 - 2*omega_0*delta_omega*II1 + delta_omega**2*II2
    )

    return phi_c, phi_c_r


def tabulated(
    path: str | Path,
    smoothing: float = 0.0,
) -> Callable:
    """Build a rotation law from a tabulated cylindrical profile.

    The first two columns contain cylindrical radius and angular
    velocity. Repeated radii are reduced to their first occurrence.
    """
    data = np.loadtxt(path, ndmin=2)

    if data.shape[1] < 2:
        raise ValueError(
            "A tabulated rotation law requires at least two columns."
        )

    _, indices = np.unique(data[:, 0], return_index=True)
    s_data, omega_data = data[indices, :2].T

    omega_eval = interpolate_func(
        s_data,
        omega_data,
        der=0,
        s=smoothing,
    )
    omega_s_eval = interpolate_func(
        s_data,
        omega_data,
        der=1,
        s=smoothing,
    )
    phi_c_eval = interpolate_func(
        s_data,
        -s_data * omega_data**2,
        der=-1,
        s=smoothing,
    )
    phi_c_s_eval = interpolate_func(
        s_data,
        -s_data * omega_data**2,
        der=0,
        s=smoothing,
    )

    def rotation_law(
        r: ArrayLike,
        t: ArrayLike,
        omega_eq: float,
        return_profile: bool = False,
        return_dprofile: bool = False,
    ):
        """Evaluate the interpolated rotation law."""
        sin_theta = (1 - t**2)**0.5
        s = r * sin_theta
        norm = omega_eq / omega_data[-1]

        if return_profile:
            omega = omega_eval(s) * norm

            if return_dprofile:
                omega_s = omega_s_eval(s) * norm
                return omega, omega_s

            return omega

        phi_c   = phi_c_eval(s)   * norm**2
        phi_c_r = phi_c_s_eval(s) * norm**2 * sin_theta

        return phi_c, phi_c_r

    return rotation_law