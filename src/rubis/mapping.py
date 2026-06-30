"""Geometrical mappings and reciprocal interpolation domains."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import roots_legendre

from .legendre import pl_eval_2D, pl_project_2D
from .numerical import interpolate_func


__all__ = [
    "MappingDerivatives",
    "ExtendedMappingDerivatives",
    "MappingGeometry",
    "compute_mapping_derivatives",
    "compute_mapping_geometry",
    "extend_mapping",
    "initialize_mapping",
    "valid_reciprocal_domain",
]


FloatArray = NDArray[np.floating]


@dataclass(kw_only=True)
class MappingDerivatives:
    """Derivatives of r2d(z, t), with z = zeta and t = cos(theta)."""

    r_t: FloatArray
    r_tt: FloatArray

    r_z: FloatArray
    r_zt: FloatArray
    r_ztt: FloatArray
    r_zz: FloatArray


@dataclass(kw_only=True)
class ExtendedMappingDerivatives:
    """Derivatives available on the mapping extended into the vacuum."""

    r_t: FloatArray
    r_tt: FloatArray
    r_z: FloatArray


@dataclass(kw_only=True)
class MappingGeometry:
    """Metric and differential terms induced by r2d(z, t)."""

    sf: FloatArray
    c2: FloatArray
    cs: FloatArray

    gzz: FloatArray
    gzt: FloatArray
    gtt: FloatArray
    gg: FloatArray

    divz: FloatArray
    divt: FloatArray
    divrelz: FloatArray
    divrelt: FloatArray

    jacobian: FloatArray


def initialize_mapping(r1d, M):
    """Initialize r2d(z, t) on M Gauss-Legendre nodes."""
    t, _ = roots_legendre(M)
    r2d = np.repeat(r1d[:, None], M, axis=1)
    return r2d, t


def valid_reciprocal_domain(x, df, safety=1.0e-4):
    """Return the domain where reciprocal interpolation remains valid."""
    df = np.atleast_2d(df.T).T
    valid = np.ones_like(df, dtype='bool')
    idx = np.arange(len(x))
    for k, dpk in enumerate(df.T) :
        idx_max = len(idx)
        condition = (dpk < safety) & (x > safety)
        if np.any(condition) : 
            idx_max = np.argwhere(condition).min()
        valid[:, k] = (idx < idx_max) & (x > safety)
    valid = np.squeeze(valid)
    return valid


def _differentiate_mapping_radially(
    r2d,
    z,
    domain_ranges,
    derivative,
    spline_order,
):
    """Differentiate r2d independently in each radial domain."""
    return np.array([
        np.hstack([
            interpolate_func(z[D], rk[D], der=derivative, k=spline_order)(z[D])
            for D in domain_ranges
        ])
        for rk in r2d.T
    ]).T
    

def compute_mapping_derivatives(
    r2d,
    z,
    t,
    max_degree,
    spline_order,
    domain_ranges,
):
    """Compute radial and angular derivatives of r2d(z, t)."""
    r_l = pl_project_2D(r2d, max_degree)
    _, r_t, r_tt = pl_eval_2D(r_l, t, der=2)

    r_z = _differentiate_mapping_radially(
        r2d,
        z,
        domain_ranges,
        derivative=1,
        spline_order=spline_order,
    )

    r_z_l = pl_project_2D(r_z, max_degree)
    _, r_zt, r_ztt = pl_eval_2D(r_z_l, t, der=2)

    r_zz = _differentiate_mapping_radially(
        r2d,
        z,
        domain_ranges,
        derivative=2,
        spline_order=spline_order,
    )

    return MappingDerivatives(
        r_t=r_t,
        r_tt=r_tt,
        r_z=r_z,
        r_zt=r_zt,
        r_ztt=r_ztt,
        r_zz=r_zz,
    )
    
    
def compute_mapping_geometry(r2d, der, t):
    """Compute metric and differential terms induced by r2d(z, t).

    Parameters
    ----------
    r2d : ndarray, shape (N, M)
        Radius of each mapped surface.
    der : MappingDerivatives
        Derivatives of r2d with respect to z and t.
    t : ndarray, shape (M,)
        Angular coordinate, with t = cos(theta).

    Returns
    -------
    MappingGeometry
        Metric coefficients, divergences, and Jacobian terms.
    """
    q = 1 - t**2
    sqrt_q = np.sqrt(q)

    # Angular part of the spherical Laplacian applied to r2d.
    S = q * der.r_tt - 2 * t * der.r_t

    # The NaNs at r = 0 are intentional: the corresponding expressions
    # are singular in these coordinates and are not evaluated directly.
    with np.errstate(all="ignore"):
        sf = np.where(
            r2d == 0.0,
            np.nan,
            1.0 / (r2d**2 + q * der.r_t**2),
        )

        c2 = r2d**2 * sf
        cs = sqrt_q * r2d * der.r_t * sf

        gzz = 1.0 / (der.r_z**2 * c2)

        gzt = np.where(
            r2d == 0.0,
            np.nan,
            sqrt_q * der.r_t / (der.r_z * r2d**2),
        )

        gtt = np.where(
            r2d == 0.0,
            np.nan,
            r2d**-2,
        )

        gg = -q * der.r_z * der.r_t * sf

        common = (
            2 * r2d
            + 2 * q * der.r_t * der.r_zt / der.r_z
            - S
        )

        divz = (
            np.where(
                r2d == 0.0,
                np.nan,
                common / (der.r_z * r2d**2),
            )
            - gzz * der.r_zz / der.r_z
        )

        divt = t * gtt / sqrt_q

        divrelz = (
            common * der.r_z * sf
            - der.r_zz / der.r_z
        )

        divrelt = t / sqrt_q * np.ones_like(r2d)

        jacobian = (
            2 * der.r_z * der.r_t
            * (
                t
                + (
                    r2d
                    - t * der.r_t
                    + q * der.r_tt
                )
                * q * der.r_t * sf
            )
            - q * (
                der.r_t * der.r_zt
                + der.r_z * der.r_tt
            )
        ) * sf

    return MappingGeometry(
        sf=sf,
        c2=c2,
        cs=cs,
        gzz=gzz,
        gzt=gzt,
        gtt=gtt,
        gg=gg,
        divz=divz,
        divt=divt,
        divrelz=divrelz,
        divrelt=divrelt,
        jacobian=jacobian,
    )


def extend_mapping(r2d, der, z_ext, extension_degree=3):
    """Extend r2d and its available derivatives into the vacuum domain."""
    z_ext = z_ext[:, None]

    surf = r2d[-1]
    surf_t = der.r_t[-1]
    surf_tt = der.r_tt[-1]

    power = (2 - z_ext)**extension_degree
    power_z = extension_degree * (2 - z_ext)**(extension_degree - 1)

    r2d_ext = np.vstack((
        r2d,
        z_ext - (1 - surf) * power,
    ))

    der_ext = ExtendedMappingDerivatives(
        r_t=np.vstack((
            der.r_t,
            surf_t * power,
        )),
        r_tt=np.vstack((
            der.r_tt,
            surf_tt * power,
        )),
        r_z=np.vstack((
            der.r_z,
            1 + (1 - surf) * power_z,
        )),
    )

    return r2d_ext, der_ext