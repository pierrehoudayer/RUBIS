import numpy as np
import pytest

from helpers import DotDict
from model_deform_radial import radial_method
from rotation_profiles import solid


def output_options():
    return DotDict(
        show_harmonics=False,
        virial_test=False,
        show_model=False,
        gravitational_moments=False,
        save_model=False,
    )


@pytest.mark.slow
@pytest.mark.readme
def test_readme_near_critical_n3_model_converges():
    mapping_precision = 1.0e-10
    rotation_target = 0.9999

    model = DotDict(
        indices=3.0,
        target_pressures=-np.inf,
        density_jumps=None,
        radius=1.0,
        mass=1.0,
        resolution=1001,
    )

    result = radial_method(
        model,
        solid,
        rotation_target,
        0.0,
        1.0,
        401,
        401,
        3,
        mapping_precision,
        5,
        3,
        output_options(),
        21,
        True,
        max_iterations=150,
    )

    assert result.iterations < 150

    assert np.isfinite(result.mapping).all()
    assert np.isfinite(result.gravitational_potential_harmonics).all()

    # Equatorial symmetry.
    np.testing.assert_allclose(
        result.mapping,
        result.mapping[:, ::-1],
        rtol=0.0,
        atol=1.0e-12,
    )

    equator = np.argmin(
        np.abs(result.cos_theta)
    )

    np.testing.assert_allclose(
        result.cos_theta[equator],
        0.0,
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        result.mapping[-1, equator],
        1.0,
        rtol=0.0,
        atol=1.0e-10,
    )

    polar_radius = (
        result.polar_radius_history[-1]
    )

    # Broad physical bounds rather than a stored reference.
    assert 0.60 < polar_radius < 0.75

    # Material surfaces remain nested.
    assert np.min(
        np.diff(result.mapping, axis=0)
    ) >= -1.0e-11

    # The actual historical stopping criterion is met.
    assert (
        abs(
            result.polar_radius_history[-1]
            - result.polar_radius_history[-2]
        )
        <= mapping_precision
    )

    np.testing.assert_allclose(
        result.rotation_rate,
        rotation_target,
        rtol=1.0e-8,
        atol=1.0e-10,
    )