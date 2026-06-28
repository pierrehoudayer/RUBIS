"""Utilities for multidomain coordinates and interfaces."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


__all__ = [
    "DomainLayout",
    "find_domains",
]


from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]
BoolArray = NDArray[np.bool_]


@dataclass
class DomainLayout:
    """Index layout of a coordinate split into adjacent domains.

    A duplicated coordinate value represents an interface. The first
    copy ends the lower domain and the second copy starts the upper
    domain.
    """

    interface_values: FloatArray
    interface_indices: tuple[IntArray, ...]
    interface_end_indices: IntArray
    interface_start_indices: IntArray

    unique_indices: IntArray
    n_domains: int

    domain_edges: IntArray
    domain_ranges: tuple[range, ...]
    domain_sizes: IntArray
    domain_index: IntArray
    domain_ids: IntArray

    internal_mask: BoolArray
    external_mask: BoolArray
    unique_internal_indices: IntArray

    @property
    def has_interfaces(self) -> bool:
        return self.interface_values.size > 0


def find_domains(coordinate) -> DomainLayout:
    """Identify domains separated by duplicated coordinate values."""
    coordinate = np.asarray(coordinate)
    n_points = coordinate.size

    (
        unique_values,
        unique_indices,
        unique_inverse,
        unique_counts,
    ) = np.unique(
        np.round(coordinate, 15),
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )

    repeated = unique_counts > 1
    interface_values = unique_values[repeated]

    repeated_value_indices = np.flatnonzero(repeated)
    interface_mask = np.isin(
        unique_inverse,
        repeated_value_indices,
    )
    flat_interface_indices = np.flatnonzero(
        interface_mask
    )

    order = np.argsort(
        unique_inverse[interface_mask]
    )

    if interface_values.size:
        interface_indices = tuple(
            np.split(
                flat_interface_indices[order],
                np.cumsum(unique_counts[repeated])[:-1],
            )
        )

        interface_array = np.asarray(
            interface_indices,
            dtype=int,
        )

        interface_end_indices = interface_array[:, 0]
        interface_start_indices = interface_array[:, 1]
    else:
        interface_indices = ()
        interface_end_indices = np.empty(
            0,
            dtype=int,
        )
        interface_start_indices = np.empty(
            0,
            dtype=int,
        )

    n_domains = interface_values.size + 1

    domain_edges = np.array(
        (
            0,
            *interface_start_indices,
            n_points,
        ),
        dtype=int,
    )

    domain_ranges = tuple(
        range(start, stop)
        for start, stop in zip(
            domain_edges[:-1],
            domain_edges[1:],
        )
    )

    domain_sizes = np.diff(domain_edges)

    domain_ids = np.arange(
        n_domains,
        dtype=int,
    )
    domain_index = np.repeat(
        domain_ids,
        domain_sizes,
    )

    external_mask = (
        domain_index == domain_ids[-1]
    )
    internal_mask = ~external_mask

    unique_internal_indices = np.unique(
        coordinate[internal_mask],
        return_index=True,
    )[1]

    return DomainLayout(
        interface_values=interface_values,
        interface_indices=interface_indices,
        interface_end_indices=interface_end_indices,
        interface_start_indices=interface_start_indices,
        unique_indices=unique_indices,
        n_domains=n_domains,
        domain_edges=domain_edges,
        domain_ranges=domain_ranges,
        domain_sizes=domain_sizes,
        domain_index=domain_index,
        domain_ids=domain_ids,
        internal_mask=internal_mask,
        external_mask=external_mask,
        unique_internal_indices=unique_internal_indices,
    )