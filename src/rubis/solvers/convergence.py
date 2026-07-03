from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ConvergenceTracker:
    """Track and optionally report convergence of a scalar quantity."""

    solver_name: str
    quantity_name: str

    tolerance: float
    max_iterations: int
    verbose: bool

    history: list[float]

    def __post_init__(self):
        if self.tolerance <= 0.0:
            raise ValueError(
                "tolerance must be positive."
            )

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
        quantity_name,
        tolerance,
        max_iterations,
        verbose=False,
    ):
        """Initialize tracking from the undeformed model."""
        tracker = cls(
            solver_name=solver_name,
            quantity_name=quantity_name,
            tolerance=tolerance,
            max_iterations=max_iterations,
            verbose=verbose,
            history=[
                float(initial_value),
            ],
        )

        tracker._report_start()

        return tracker

    @property
    def iterations(self):
        """Number of completed solver iterations."""
        return len(self.history) - 1

    @property
    def current(self):
        """Most recently recorded value."""
        return self.history[-1]

    @property
    def error(self):
        """Absolute change between the two latest values."""
        if self.iterations == 0:
            return np.inf

        return abs(
            self.history[-1]
            - self.history[-2]
        )

    @property
    def converged(self):
        """Whether an iteration satisfies the tolerance."""
        return (
            self.iterations > 0
            and self.error <= self.tolerance
        )

    def update(self, value):
        """Record and optionally report a completed iteration."""
        self.history.append(
            float(value)
        )

        self._report_iteration()

    def check_iteration_limit(self):
        """Raise when no additional iteration is allowed."""
        if self.iterations < self.max_iterations:
            return

        recent_values = np.asarray(
            self.history[-4:]
        )

        iteration_word = (
            "iteration"
            if self.max_iterations == 1
            else "iterations"
        )

        raise RuntimeError(
            f"{self.solver_name} deformation did not converge "
            f"after {self.max_iterations} {iteration_word}. "
            f"Last {self.quantity_name} change = "
            f"{self.error:.3e}, "
            f"target = {self.tolerance:.3e}. "
            f"Recent values: {recent_values!r}"
        )

    def report_convergence(
        self,
        *,
        elapsed_time=None,
    ):
        """Report the final convergence state."""
        if not self.verbose:
            return

        iteration_word = (
            "iteration"
            if self.iterations == 1
            else "iterations"
        )

        message = (
            f"  converged after "
            f"{self.iterations} {iteration_word}: "
            f"error = {self.error:.3e}"
        )

        if elapsed_time is not None:
            message += (
                f", elapsed time = "
                f"{elapsed_time:.2f} s"
            )

        print(message)

    def _report_start(self):
        if not self.verbose:
            return

        print(
            f"{self.solver_name} deformation"
        )
        print(
            f"  initial {self.quantity_name} "
            f"= {self.current:.12g}"
        )

    def _report_iteration(self):
        if not self.verbose:
            return

        print(
            f"  iteration {self.iterations:02d}: "
            f"{self.quantity_name} "
            f"= {self.current:.12g}, "
            f"error = {self.error:.3e}"
        )