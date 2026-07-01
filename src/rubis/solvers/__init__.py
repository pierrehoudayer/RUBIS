"""Centrifugal-deformation solvers."""

from .radial import solve_radial
from .spheroidal import solve_spheroidal


__all__ = [
    "solve_radial",
    "solve_spheroidal",
]