"""Input and output for the historical RUBIS text format."""

from pathlib import Path

import numpy as np


__all__ = ["write_model"]


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