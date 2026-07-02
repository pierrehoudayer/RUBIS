from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ConvergenceTracker:
    """
    Track the convergence of a scalar solver quantity.

    The initial history preserves the legacy zero reference, while each
    subsequent value corresponds to one completed solver iteration.
    """

    solver_name: str
    tolerance: float
    max_iterations: int
    history: list[float]

    def __post_init__(self):
        if self.max_iterations < 1:
            raise ValueError(
                "max_iterations must be a positive integer."
            )

    @classmethod
    def start(
        cls,
        initial_value,
        *,
        solver_name,
        tolerance,
        max_iterations,
    ):
        """Initialize convergence tracking from the undeformed model."""
        return cls(
            solver_name=solver_name,
            tolerance=tolerance,
            max_iterations=max_iterations,
            history=[0.0, initial_value],
        )

    @property
    def iterations(self):
        """Number of completed solver iterations."""
        return len(self.history) - 2

    @property
    def current(self):
        """Most recently recorded value."""
        return self.history[-1]

    @property
    def error(self):
        """Absolute change between the two latest values."""
        return abs(
            self.history[-1]
            - self.history[-2]
        )

    @property
    def converged(self):
        """Whether the requested tolerance is satisfied."""
        return self.error <= self.tolerance

    def update(self, value):
        """Record the value produced by one completed iteration."""
        self.history.append(value)

    def check_iteration_limit(self):
        """Raise when no additional iteration is allowed."""
        if self.iterations < self.max_iterations:
            return

        recent_values = np.asarray(
            self.history[-4:]
        )

        raise RuntimeError(
            f"{self.solver_name} deformation did not converge "
            f"after {self.max_iterations} iterations. "
            f"Last |delta R_pol| = {self.error:.3e}, "
            f"target = {self.tolerance:.3e}. "
            f"Recent polar radii: {recent_values!r}"
        )