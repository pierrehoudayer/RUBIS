"""Utilities for multidomain coordinates and interfaces."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


__all__ = [
    "DomainLayout",
    "find_domains",
]


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


@dataclass(frozen=True, slots=True)
class DomainLayout:
    """
    Index layout of a coordinate split into adjacent domains.

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

    @property
    def n_points(self) -> int:
        return self.domain_index.size

    @property
    def has_interfaces(self) -> bool:
        return self.interface_values.size > 0


def find_domains(coordinate) -> DomainLayout:
    """Identify domains separated by duplicated coordinate values."""
    coordinate = np.asarray(coordinate)

    if coordinate.ndim != 1:
        raise ValueError(
            "The domain coordinate must be one-dimensional."
        )

    if coordinate.size == 0:
        raise ValueError(
            "The domain coordinate must not be empty."
        )

    rounded = np.round(
        coordinate,
        15,
    )

    if np.any(np.diff(rounded) < 0.0):
        raise ValueError(
            "The domain coordinate must be non-decreasing."
        )

    (
        unique_values,
        unique_indices,
        unique_inverse,
        unique_counts,
    ) = np.unique(
        rounded,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )

    if np.any(unique_counts > 2):
        raise ValueError(
            "Each domain interface must occur exactly twice."
        )

    repeated = unique_counts == 2
    repeated_value_indices = np.flatnonzero(
        repeated
    )

    interface_values = unique_values[
        repeated
    ]
    interface_indices = tuple(
        np.flatnonzero(
            unique_inverse == value_index
        )
        for value_index
        in repeated_value_indices
    )

    interface_end_indices = np.array(
        [
            indices[0]
            for indices in interface_indices
        ],
        dtype=int,
    )
    interface_start_indices = np.array(
        [
            indices[1]
            for indices in interface_indices
        ],
        dtype=int,
    )

    n_domains = (
        interface_values.size + 1
    )

    domain_edges = np.array(
        (
            0,
            *interface_start_indices,
            coordinate.size,
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

    domain_sizes = np.diff(
        domain_edges
    )
    domain_ids = np.arange(
        n_domains,
        dtype=int,
    )
    domain_index = np.repeat(
        domain_ids,
        domain_sizes,
    )

    return DomainLayout(
        interface_values=interface_values,
        interface_indices=interface_indices,
        interface_end_indices=(
            interface_end_indices
        ),
        interface_start_indices=(
            interface_start_indices
        ),
        unique_indices=unique_indices,
        n_domains=n_domains,
        domain_edges=domain_edges,
        domain_ranges=domain_ranges,
        domain_sizes=domain_sizes,
        domain_index=domain_index,
        domain_ids=domain_ids,
    )