import numpy as np
import pytest

from helpers import DotDict, assign_method


def radial_solver():
    pass


def spheroidal_solver():
    pass


@pytest.mark.parametrize(
    ("indices", "expected_method"),
    [
        (3.0, radial_solver),
        ((3.0,), radial_solver),
        ((1.0, 1.0), spheroidal_solver),
    ],
)
def test_auto_method_selection_for_polytropes(
    indices,
    expected_method,
):
    model = DotDict(
        indices=indices,
        target_pressures=-np.inf,
        density_jumps=None,
        radius=1.0,
        mass=1.0,
        resolution=65,
    )

    method = assign_method(
        "auto",
        model,
        radial_solver,
        spheroidal_solver,
    )

    assert method is expected_method


@pytest.mark.parametrize(
    ("method_choice", "expected_method"),
    [
        ("radial", radial_solver),
        ("spheroidal", spheroidal_solver),
    ],
)
def test_explicit_method_selection(
    method_choice,
    expected_method,
):
    model = DotDict(
        indices=(1.0, 1.0),
        target_pressures=(-1.0, -np.inf),
        density_jumps=(0.4,),
        radius=1.0,
        mass=1.0,
        resolution=65,
    )

    method = assign_method(
        method_choice,
        model,
        radial_solver,
        spheroidal_solver,
    )

    assert method is expected_method