"""B-spline interpolation helpers."""

from collections.abc import Sequence

import numpy as np
from scipy.interpolate import splantider, splev, splrep


__all__ = [
    "interpolate_func",
]


def interpolate_func(
    x,
    y,
    der: int | Sequence[int] = 0,
    k: int = 3,
    prim_cond=None,
    *args,
    **kwargs,
):
    """Build a B-spline evaluator, derivative, or antiderivative."""
    tck = splrep(
        x,
        y,
        *args,
        k=k,
        **kwargs,
    )

    if not isinstance(der, (int, np.integer)):
        derivatives = tuple(der)

        if any(
            not isinstance(d, (int, np.integer))
            or not 0 <= d < k
            for d in derivatives
        ):
            raise ValueError(
                "Derivative orders must satisfy 0 <= der < k."
            )

        def evaluate(x_eval):
            x_eval = np.asarray(x_eval)

            if 0 in x_eval.shape:
                return np.empty((len(derivatives), 0))

            return [
                splev(x_eval, tck, der=int(d))
                for d in derivatives
            ]

        return evaluate

    der = int(der)

    if not -2 < der < k:
        raise ValueError(
            "Derivative order must be -1 or satisfy "
            f"0 <= der < k={k}; got der={der}."
        )

    if der >= 0:
        def evaluate(x_eval):
            x_eval = np.asarray(x_eval)

            if 0 in x_eval.shape:
                return np.array([])

            return splev(x_eval, tck, der=der)

        return evaluate

    tck_antiderivative = splantider(tck)
    constant = 0.0

    if prim_cond is not None:
        index, value = prim_cond
        constant = (
            value
            - splev(x[index], tck_antiderivative)
        )

    def evaluate(x_eval):
        return (
            splev(x_eval, tck_antiderivative)
            + constant
        )

    return evaluate