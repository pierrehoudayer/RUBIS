"""Input and output for the historical RUBIS text format."""

from pathlib import Path

import numpy as np

from ..models import PolytropicModelConfig


__all__ = [
    "make_output_filename",
    "write_model",
]


def make_output_filename(model_choice, rotation_target):
    """Build the output filename for a deformed model."""
    if isinstance(model_choice, PolytropicModelConfig):
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