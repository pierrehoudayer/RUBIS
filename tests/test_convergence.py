import pytest

from rubis.solvers.convergence import (
    ConvergenceTracker,
)


def test_convergence_tracker_records_iterations():
    conv = ConvergenceTracker.start(
        1.0,
        solver_name="Radial",
        tolerance=1.0e-3,
        max_iterations=3,
    )

    assert conv.iterations == 0
    assert conv.current == 1.0
    assert not conv.converged

    conv.update(0.9)

    assert conv.iterations == 1
    assert conv.current == 0.9
    assert conv.error == pytest.approx(0.1)
    assert not conv.converged

    conv.update(0.9005)

    assert conv.iterations == 2
    assert conv.error == pytest.approx(5.0e-4)
    assert conv.converged
    
    
def test_convergence_tracker_enforces_iteration_limit():
    conv = ConvergenceTracker.start(
        1.0,
        solver_name="Radial",
        tolerance=1.0e-10,
        max_iterations=1,
    )

    conv.check_iteration_limit()
    conv.update(0.9)

    with pytest.raises(
        RuntimeError,
        match="did not converge after 1 iterations",
    ):
        conv.check_iteration_limit()
        
        
def test_convergence_tracker_requires_positive_limit():
    with pytest.raises(
        ValueError,
        match="max_iterations must be a positive integer",
    ):
        ConvergenceTracker.start(
            1.0,
            solver_name="Radial",
            tolerance=1.0e-10,
            max_iterations=0,
        )