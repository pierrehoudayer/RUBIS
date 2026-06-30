"""Public interface for centrifugal-deformation calculations."""

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
        config.rotation,
        config.solver,
        config.output,
    )