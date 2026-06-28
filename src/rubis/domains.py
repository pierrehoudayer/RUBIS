"""Utilities for multidomain coordinates and interfaces."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


__all__ = [
    "DomainLayout",
    "find_domains",
]


FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]
BoolArray = NDArray[np.bool_]


@dataclass
class DomainLayout:
    """Layout of a coordinate split into adjacent domains.

    A duplicated coordinate value represents an interface. Its first
    occurrence belongs to the lower domain and its second occurrence
    belongs to the upper domain.
    """

    bounds: FloatArray
    interfaces: list[IntArray]

    # None for a continuous, single-domain coordinate.
    end: IntArray | None
    beg: IntArray | None

    unq: IntArray
    Nd: int
    edges: IntArray
    ranges: list[range]
    sizes: list[int]

    id: NDArray[np.floating]
    id_val: NDArray[np.floating]

    ext: BoolArray
    int: BoolArray
    unq_int: IntArray


def find_domains(var) -> DomainLayout:
    """Identify domains separated by duplicated coordinate values.

    Parameters
    ----------
    var : array_like, shape (N,)
        Coordinate used to define the domains. A duplicated value
        represents an interface shared by two adjacent domains.

    Returns
    -------
    DomainLayout
        Domain layout and navigation information.
    """
    var = np.asarray(var)
    n_var = len(var)

    unique, unique_indices, unique_inverse, unique_counts = np.unique(
        np.round(var, 15),
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )

    repeated = unique_counts > 1
    bounds = unique[repeated]
    discontinuous = bounds.size > 0

    repeated_indices, = np.nonzero(repeated)
    interface_mask = np.isin(
        unique_inverse,
        repeated_indices,
    )
    interface_indices, = np.nonzero(interface_mask)

    order = np.argsort(
        unique_inverse[interface_mask]
    )

    interfaces = np.split(
        interface_indices[order],
        np.cumsum(unique_counts[repeated])[:-1],
    )

    if discontinuous:
        end, beg = np.asarray(interfaces).T
    else:
        end = None
        beg = None

    unq = unique_indices
    number_of_domains = len(bounds) + 1

    if discontinuous:
        edges = np.array(
            (0, *beg, n_var)
        )
    else:
        edges = np.array(
            (0, n_var)
        )

    ranges = [
        range(start, stop)
        for start, stop in zip(
            edges[:-1],
            edges[1:],
        )
    ]
    sizes = [
        len(domain_range)
        for domain_range in ranges
    ]

    domain_id = np.hstack([
        domain * np.ones(size)
        for domain, size in enumerate(sizes)
    ])
    domain_id_values = np.unique(domain_id)

    external_mask = (
        domain_id == number_of_domains - 1
    )
    internal_mask = ~external_mask

    unique_internal_indices = np.unique(
        var[internal_mask],
        return_index=True,
    )[1]

    return DomainLayout(
        bounds=bounds,
        interfaces=interfaces,
        end=end,
        beg=beg,
        unq=unq,
        Nd=number_of_domains,
        edges=edges,
        ranges=ranges,
        sizes=sizes,
        id=domain_id,
        id_val=domain_id_values,
        ext=external_mask,
        int=internal_mask,
        unq_int=unique_internal_indices,
    )