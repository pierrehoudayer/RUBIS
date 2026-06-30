"""Public interface for centrifugal-deformation calculations."""

from pathlib import Path

import numpy as np

from .config import DeformationConfig, SolverMethod
from .domains import find_domains
from .models import PolytropicModelConfig
from .results import DeformationResult
from .solvers import radial_method, spheroidal_method


__all__ = [
    "deform",
]


def _select_method(config: DeformationConfig) -> SolverMethod:
    """Select the deformation method requested by a configuration."""
    method = config.solver.method

    if method != "auto":
        if method not in ("radial", "spheroidal"):
            raise ValueError(
                f"Unknown deformation method {method!r}; expected "
                "'auto', 'radial' or 'spheroidal'."
            )

        return method

    model = config.model

    if isinstance(model, PolytropicModelConfig):
        n_domains = model.n_regions
    else:
        r1d = np.genfromtxt(
            Path("Models") / model,
            skip_header=2,
            usecols=0,
        )
        n_domains = find_domains(r1d).n_domains

    if n_domains > 1:
        return "spheroidal"

    return "radial"


def deform(config: DeformationConfig) -> DeformationResult:
    """Compute the centrifugal deformation specified by config."""
    method = _select_method(config)

    solvers = {
        "radial": radial_method,
        "spheroidal": spheroidal_method,
    }
    solver = solvers[method]

    return solver(
        config.model,
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