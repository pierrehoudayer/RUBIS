"""Public interface for centrifugal-deformation calculations."""

from pathlib import Path

import numpy as np

from .config import (
    DeformationConfig,
    SolverMethod,
)
from .initialization import initialize_model_1d
from .models import Model1D
from .results import DeformationResult
from .solvers import radial_method, spheroidal_method


__all__ = [
    "deform",
]


def _select_method(
    method: SolverMethod,
    model: Model1D,
) -> SolverMethod:
    if method not in ("auto", "radial", "spheroidal"):
        raise ValueError(
            f"Unknown deformation method {method!r}; expected "
            "'auto', 'radial' or 'spheroidal'."
        )

    if method != "auto":
        return method

    return (
        "spheroidal"
        if model.n_domains > 1
        else "radial"
    )


def deform(config: DeformationConfig) -> DeformationResult:
    """Compute the centrifugal deformation specified by config."""
    model = initialize_model_1d(config.model)

    method = _select_method(
        config.solver.method,
        model,
    )

    solver = {
        "radial": radial_method,
        "spheroidal": spheroidal_method,
    }[method]

    return solver(
        model,
        config.rotation.profile,
        config.rotation.target,
        config.rotation.central_diff_rate,
        config.rotation.scale,
        config.solver.max_degree,
        config.solver.angular_resolution,
        config.solver.full_rate,
        config.solver.mapping_precision,
        config.solver.spline_order,
        config.solver.lagrange_order,
        config.output,
        config.solver.external_domain_res,
        config.solver.rescale_ab,
        max_iterations=config.solver.max_iterations,
    )