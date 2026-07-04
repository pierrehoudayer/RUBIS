"""Information returned alongside RUBIS model solutions."""

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from .models import Model2D, VacuumModel2D


FloatArray = NDArray[np.floating]

__all__ = [
    "ResolvedSolverMethod",
    "SolverInfo",
    "SolverOutput",
]


ResolvedSolverMethod: TypeAlias = Literal[
    "radial",
    "spheroidal",
]


@dataclass(kw_only=True)
class SolverInfo:
    """
    Information about a completed deformation solve.

    The final physical rotation belongs to the returned model, while
    rotation_target records the continuation target used by the solver.
    """

    method: ResolvedSolverMethod

    iterations: int
    tolerance: float
    error: float
    polar_radius_history: FloatArray

    rotation_target: float
    elapsed_time: float


SolverOutput: TypeAlias = tuple[
    Model2D,
    VacuumModel2D | None,
    SolverInfo,
]