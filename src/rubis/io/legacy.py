"""Input and output for the historical RUBIS text format."""

from pathlib import Path

import numpy as np

from ..config import (
    CompositePolytropeConfig,
    PolytropeConfig, 
)
from ..models import Model2D
from ..results import SolverInfo


__all__ = [
    "make_output_filename",
    "write_deformed_model",
    "write_model",
]


def make_output_filename(model_choice, rotation_target):
    """Build the output filename for a deformed model."""
    if isinstance(model_choice, PolytropeConfig | CompositePolytropeConfig):
        model_name = model_choice.filename_stem
    else:
        model_name = str(model_choice).removesuffix(".txt")

    return f"{model_name}_deform_{rotation_target}.txt"


def write_model(filename, params, mapping, additional_variables, *variables):
    """Write a deformed model in the historical RUBIS text format.

    Parameters
    ----------
    filename : str or path-like
        Name of the output file, written inside the ``Models`` directory.
    params : iterable
        Global model parameters written on the first line.
    mapping : ndarray, shape (N, M)
        Isopotential mapping.
    additional_variables : iterable of ndarray
        Variables read from the original model and left unchanged.
    *variables : ndarray
        Computed one-dimensional variables appended after the mapping.
    """
    header = " ".join(str(value) for value in params)
    data = np.hstack((mapping, np.vstack((*variables, *additional_variables)).T))

    np.savetxt(
        Path("Models") / filename,
        data,
        header=header,
        comments="",
    )
    
    
def write_deformed_model(
    filename,
    model: Model2D,
    info: SolverInfo,
    *,
    dimensional=False,
):
    """
    Write a converged model in the historical RUBIS format.

    Dimensional output scales the computed mechanical fields while
    preserving the material coordinate, rotation profile, and
    additional variables in their existing conventions.
    """
    I, J = model.r2d.shape

    # The angular grid produced by RUBIS contains the equator
    j_eq = np.argmin(
        np.abs(model.t)
    )
    omega_equator = model.omega[:, j_eq]

    r2d = model.r2d
    rho = model.rho
    p = model.p
    phi_eff = model.phi_eff

    if dimensional:
        r2d = (
            r2d
            * model.radius
        )
        rho = (
            rho
            * model.mass
            / model.radius**3
        )
        phi_eff = (
            phi_eff
            * model.G
            * model.mass
            / model.radius
        )
        p = (
            p
            * model.G
            * model.mass**2
            / model.radius**4
        )

    write_model(
        filename,
        (
            I,
            J,
            model.mass,
            model.radius,
            info.rotation_target,
            model.G,
        ),
        r2d,
        model.additional_variables,
        model.zeta,
        p,
        rho,
        phi_eff,
        omega_equator,
    )