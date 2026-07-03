import numpy as np

from rubis.results import SolverInfo


def test_solver_info_stores_convergence_state():
    history = np.array([1.0, 0.9, 0.9001])

    info = SolverInfo(
        method="radial",
        iterations=2,
        tolerance=1.0e-3,
        error=1.0e-4,
        polar_radius_history=history,
        rotation_target=0.5,
        elapsed_time=1.2,
    )

    assert info.method == "radial"
    assert info.iterations == 2
    assert info.error < info.tolerance
    np.testing.assert_array_equal(
        info.polar_radius_history,
        history,
    )