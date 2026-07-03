import numpy as np
import pytest

from rubis.solvers.convergence import (
    ConvergenceTracker,
)


def test_convergence_tracker_records_iterations():
    conv = ConvergenceTracker.start(
        1.0,
        solver_name="Radial",
        quantity_name="R_pol",
        tolerance=1.0e-3,
        max_iterations=3,
    )

    assert conv.iterations == 0
    assert conv.current == 1.0
    assert conv.history == [1.0]
    assert np.isinf(conv.error)
    assert not conv.converged

    conv.update(0.9)

    assert conv.iterations == 1
    assert conv.current == 0.9
    assert conv.error == pytest.approx(
        0.1
    )
    assert not conv.converged

    conv.update(0.9005)

    assert conv.iterations == 2
    assert conv.error == pytest.approx(
        5.0e-4
    )
    assert conv.converged


def test_convergence_tracker_reports_progress(
    capsys,
):
    conv = ConvergenceTracker.start(
        1.0,
        solver_name="Radial",
        quantity_name="R_pol",
        tolerance=1.0e-3,
        max_iterations=3,
        verbose=True,
    )

    conv.update(0.9)
    conv.update(0.9005)

    conv.report_convergence(
        elapsed_time=1.25,
    )

    output = capsys.readouterr().out

    assert "Radial deformation" in output
    assert "initial R_pol = 1" in output
    assert "iteration 01" in output
    assert "R_pol = 0.9" in output
    assert "error = 1.000e-01" in output
    assert "converged after 2 iterations" in output
    assert "elapsed time = 1.25 s" in output


def test_convergence_tracker_is_silent_by_default(
    capsys,
):
    conv = ConvergenceTracker.start(
        1.0,
        solver_name="Radial",
        quantity_name="R_pol",
        tolerance=1.0e-3,
        max_iterations=3,
    )

    conv.update(0.9)
    conv.report_convergence(
        elapsed_time=1.0,
    )

    assert capsys.readouterr().out == ""


def test_convergence_tracker_enforces_iteration_limit():
    conv = ConvergenceTracker.start(
        1.0,
        solver_name="Radial",
        quantity_name="R_pol",
        tolerance=1.0e-10,
        max_iterations=1,
    )

    conv.check_iteration_limit()
    conv.update(0.9)

    with pytest.raises(
        RuntimeError,
        match=(
            "did not converge after "
            "1 iteration"
        ),
    ):
        conv.check_iteration_limit()


def test_convergence_tracker_requires_positive_limit():
    with pytest.raises(
        ValueError,
        match=(
            "max_iterations must be "
            "a positive integer"
        ),
    ):
        ConvergenceTracker.start(
            1.0,
            solver_name="Radial",
            quantity_name="R_pol",
            tolerance=1.0e-10,
            max_iterations=0,
        )


def test_convergence_tracker_requires_positive_tolerance():
    with pytest.raises(
        ValueError,
        match="tolerance must be positive",
    ):
        ConvergenceTracker.start(
            1.0,
            solver_name="Radial",
            quantity_name="R_pol",
            tolerance=0.0,
            max_iterations=3,
        )