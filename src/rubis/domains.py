"""Utilities for multidomain coordinates and interfaces."""

import numpy as np

from ._utils import DotDict


__all__ = ["find_domains"]


def find_domains(var):
    """Identify domains separated by duplicated coordinate values.

    Parameters
    ----------
    var : array_like, shape (N,)
        Coordinate used to define the domains. A duplicated value
        represents an interface shared by two adjacent domains.

    Returns
    -------
    dom : DotDict
        Domain layout and navigation information.
    """
    dom = DotDict()
    n_var = len(var)
    discontinuous = True

    # Physical domain boundaries
    unique, unique_indices, unique_inverse, unique_counts = np.unique(
        np.round(var, 15),
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )

    repeated = unique_counts > 1
    dom.bounds = unique[repeated]

    if len(dom.bounds) == 0:
        discontinuous = False

    # Interface indices
    repeated_indices, = np.nonzero(repeated)
    interface_mask = np.isin(
        unique_inverse,
        repeated_indices,
    )
    interface_indices, = np.nonzero(interface_mask)

    order = np.argsort(
        unique_inverse[interface_mask]
    )

    dom.interfaces = np.split(
        interface_indices[order],
        np.cumsum(unique_counts[repeated])[:-1],
    )

    if discontinuous:
        dom.end, dom.beg = np.array(
            dom.interfaces
        ).T

    # Domain ranges and sizes
    dom.unq = unique_indices
    dom.Nd = len(dom.bounds) + 1

    if discontinuous:
        dom.edges = np.array(
            (0, *dom.beg, n_var)
        )
    else:
        dom.edges = np.array(
            (0, n_var)
        )

    dom.ranges = [
        range(start, stop)
        for start, stop in zip(
            dom.edges[:-1],
            dom.edges[1:],
        )
    ]
    dom.sizes = [
        len(domain_range)
        for domain_range in dom.ranges
    ]

    # Domain identification
    dom.id = np.hstack([
        domain * np.ones(size)
        for domain, size in enumerate(dom.sizes)
    ])
    dom.id_val = np.unique(dom.id)

    dom.ext = dom.id == dom.Nd - 1
    dom.int = ~dom.ext

    dom.unq_int = np.unique(
        np.asarray(var)[dom.int],
        return_index=True,
    )[1]

    return dom