"""Geometrical mappings and reciprocal interpolation domains."""

import numpy as np
from scipy.special import roots_legendre


__all__ = [
    "initialize_mapping",
    "valid_reciprocal_domain",
]


def initialize_mapping(r, M):
    """Initialize a spherical mapping on M Gauss--Legendre nodes."""
    cth, _ = roots_legendre(M)
    mapping = np.repeat(r[:, None], M, axis=1)
    return mapping, cth


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