import numpy as np

from helpers import DotDict
from model_deform_spheroidal import spheroidal_method
from rotation_profiles import solid


def test_spheroidal_solver_returns_normalised_state():
    resolution = 65
    external_resolution = 21
    angular_resolution = 9
    max_degree = 9

    model = DotDict(
        indices=(1.0, 1.0),
        target_pressures=(-1.0, -np.inf),
        density_jumps=(0.4,),
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

    result = spheroidal_method(
        model,
        solid,
        0.0,
        0.0,
        1.0,
        max_degree,
        angular_resolution,
        1,
        1.0e-10,
        3,
        2,
        output,
        external_resolution,
        True,
    )

    assert result.mapping.shape == (
        resolution,
        angular_resolution,
    )
    assert result.full_mapping.shape == (
        resolution + external_resolution,
        angular_resolution,
    )

    assert result.zeta.shape == (
        resolution + external_resolution,
    )
    assert result.internal_zeta.shape == (
        resolution,
    )
    assert result.external_zeta.shape == (
        external_resolution,
    )

    assert result.density.shape == (
        resolution,
    )
    assert result.pressure.shape == (
        resolution,
    )

    assert (
        result.gravitational_potential_harmonics.shape
        == (
            resolution + external_resolution,
            max_degree,
        )
    )
    assert (
        result.gravitational_potential_derivative_harmonics.shape
        == (
            resolution + external_resolution,
            max_degree,
        )
    )

    assert result.internal_mask.shape == result.zeta.shape
    assert result.external_mask.shape == result.zeta.shape

    assert np.count_nonzero(
        result.internal_mask
    ) == resolution
    assert np.count_nonzero(
        result.external_mask
    ) == external_resolution

    assert np.isfinite(result.mapping).all()
    assert np.isfinite(result.full_mapping).all()
    assert np.isfinite(result.density).all()
    assert np.isfinite(result.pressure).all()
    assert np.isfinite(
        result.gravitational_potential_harmonics
    ).all()

    assert result.iterations >= 1