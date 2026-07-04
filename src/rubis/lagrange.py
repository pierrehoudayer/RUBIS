"""Lagrange interpolation and differentiation operators."""

from itertools import combinations

import numpy as np
from numpy.polynomial.polynomial import Polynomial


__all__ = [
    "lagrange_matrix_P",
]


def _find_order_two_roots(coefficients):
    """Select the required roots of the order-two stencil polynomials."""
    solve_cubic = np.ones(
        coefficients.shape[1],
        dtype=bool,
    )
    solve_cubic[0], solve_cubic[-1] = False, False

    roots = np.empty(coefficients.shape[1])
    c2 = coefficients[:, ~solve_cubic]
    c3 = coefficients[:, solve_cubic]

    discriminant = np.sqrt(
        c2[1]*c2[1]
        - 4.0*c2[2]*c2[0]
    )
    roots[0] = (
        -c2[1, 0] - discriminant[0]
    ) / (2.0*c2[2, 0])
    roots[-1] = (
        -c2[1, 1] + discriminant[1]
    ) / (2.0*c2[2, 1])

    f = (
        3.0*c3[1] / c3[3]
        - c3[2]**2 / c3[3]**2
    ) / 3.0
    g = (
           2.0*c3[2]**3    / c3[3]**3
        -  9.0*c3[2]*c3[1] / c3[3]**2
        + 27.0*c3[0]       / c3[3]
    ) / 27.0
    h = g**2 / 4.0 + f**3 / 27.0

    i = np.sqrt(g**2 / 4.0 - h)
    j = i**(1/3.0)
    angle = np.arccos(-g / (2*i))

    roots[solve_cubic] = (
        -j * (
            np.cos(angle / 3.0)
            - np.sqrt(3) * np.sin(angle / 3.0)
        )
        - c3[2] / (3.0*c3[3])
    )

    return roots


def _find_stencil_root(
    index,
    nodes,
    coefficients,
    order,
):
    """Select the polynomial root lying in the current cell."""
    roots = Polynomial(coefficients).roots()

    shift = max(order - 1 - index, 0)
    lower = nodes[order - 1 - shift]
    upper = nodes[order - shift]

    return np.squeeze(
        roots[
              (roots > lower)
            & (roots < upper)
        ]
    )


def _polynomial_coefficients(
    nodes,
    order,
):
    """Return the coefficients defining one staggered stencil point."""
    coefficients = [
        (
            sum(
                np.prod(
                    list(combinations(nodes, degree)),
                    axis=1,
                )
            )
            * (len(nodes) - degree)
            * (-1)**degree
        )
        for degree in range(
            len(nodes) - 1,
            -1,
            -1,
        )
    ]

    return (
        coefficients
        + [0.0] * (2*order - len(nodes))
    )


def lagrange_matrix_P(
    x,
    order: int = 2,
):
    """Build staggered Lagrange interpolation and derivative matrices."""
    x = np.asarray(x)
    n_points = x.size

    matrix = np.zeros((
        n_points - 1,
        n_points,
        2,
    ))

    def convolve_same(a, values):
        return np.convolve(
            a,
            values,
            mode="same",
        )

    vector_convolution = np.vectorize(
        convolve_same,
        signature="(n),(m)->(n)",
    )
    mask = vector_convolution(
        np.eye(n_points, dtype=bool),
        [True] * (2*order),
    )[:-1]

    def rescale(values, reference):
        return (
            2*values
            - (reference[-1] + reference[0])
        ) / (
            reference[-1] - reference[0]
        )

    nodes = [
        rescale(x[mask_i], x[mask_i])
        for mask_i in mask
    ]
    scales = np.array([
        0.5 * (
            x[mask_i][-1]
            - x[mask_i][0]
        )
        for mask_i in mask
    ])
    coefficients = np.array([
        _polynomial_coefficients(
            nodes_i,
            order,
        )
        for nodes_i in nodes
    ])

    if order == 2:
        roots = _find_order_two_roots(
            coefficients.T
        )
    else:
        roots = [
            _find_stencil_root(
                i,
                nodes_i,
                coefficients_i,
                order,
            )
            for i, (
                nodes_i,
                coefficients_i,
            ) in enumerate(zip(
                nodes,
                coefficients,
            ))
        ]

    for i, (
        nodes_i,
        root_i,
        mask_i,
    ) in enumerate(zip(
        nodes,
        roots,
        mask,
    )):
        n_nodes = len(nodes_i)
        other_nodes = np.tile(
            nodes_i,
            n_nodes,
        )[
            ([False] + [True] * n_nodes) * (n_nodes - 1) + [False]
        ].reshape((n_nodes, -1))

        lagrange = np.prod(
            (
                root_i - other_nodes
            ) / (
                nodes_i[:, None] - other_nodes
            ),
            axis=1,
        )
        derivative = np.sum(
            lagrange[:, None]
            / (root_i - other_nodes),
            axis=1,
        )

        matrix[i, mask_i, 0] = lagrange
        matrix[i, mask_i, 1] = derivative

    matrix[..., 1] /= scales[:, None]

    return matrix