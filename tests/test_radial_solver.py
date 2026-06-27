import numpy as np

from helpers import DotDict
from model_deform_radial import radial_method
from rotation_profiles import solid


def test_radial_solver_returns_normalised_state():
    resolution = 65
    angular_resolution = 9
    max_degree = 9

    model = DotDict(
        indices=1.0,
        target_pressures=-np.inf,
        density_jumps=None,
        radius=1.0,
        mass=1.0,
        resolution=resolution,
    )

    output = DotDict(
        show_harmonics=False,
        virial_test=False,
        show_model=False,
        gravitational_moments=False,
        save_model=False,
    )

    result = radial_method(
        model,
        solid,
        0.0,    # Target rotation
        0.0,    # Central differential rotation
        1.0,    # Rotation scale
        max_degree,
        angular_resolution,
        1,      # Rotation ramp
        1.0e-10,
        3,      # Spline order
        2,      # Lagrange order
        output,
        21,     # Unused external-domain resolution
        True,   # Unused matrix rescaling option
    )

    assert result.mapping.shape == (
        resolution,
        angular_resolution,
    )
    assert result.gravitational_potential_harmonics.shape == (
        resolution,
        max_degree,
    )
    assert result.cos_theta.shape == (
        angular_resolution,
    )
    assert result.polar_radius_history.shape == (
        result.iterations + 2,
    )

    assert np.isfinite(result.mapping).all()
    assert np.isfinite(result.density).all()
    assert np.isfinite(result.pressure).all()
    assert np.isfinite(
        result.gravitational_potential_harmonics
    ).all()

    assert result.iterations >= 1