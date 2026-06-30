from pathlib import Path

import numpy as np

from rubis.options import OutputOptions
from rubis.rotation_profiles import solid
from rubis.models import CompositePolytropeConfig
from rubis.results import DeformationResult, SpheroidalResult
from rubis.solvers import spheroidal_method


def test_spheroidal_solver_returns_normalised_state():
    resolution = 65
    external_resolution = 21
    angular_resolution = 9
    max_degree = 9

    model = CompositePolytropeConfig(
        indices=(1.0, 1.0),
        target_pressures=(-1.0, -np.inf),
        density_jumps=(0.4,),
        radius=1.0,
        mass=1.0,
        n_points=resolution,
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
        OutputOptions(),
        external_resolution,
        True,
    )

    assert isinstance(result, SpheroidalResult)
    assert isinstance(result, DeformationResult)
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
    assert result.polar_radius_history.ndim == 1
    assert isinstance(result.iterations, int)

    assert result.iterations >= 1
    
    
def test_nonrotating_composite_model_remains_spherical():
    resolution = 65
    external_resolution = 21
    angular_resolution = 9
    max_degree = 9

    model = CompositePolytropeConfig(
        indices=(1.0, 1.0),
        target_pressures=(-1.0, -np.inf),
        density_jumps=(0.4,),
        radius=1.0,
        mass=1.0,
        n_points=resolution,
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
        OutputOptions(),
        external_resolution,
        True,
    )

    # Every material surface must be spherical.
    np.testing.assert_allclose(
        np.ptp(result.mapping, axis=1),
        0.0,
        rtol=0.0,
        atol=1.0e-11,
    )

    radial_mapping = np.mean(
        result.mapping,
        axis=1,
    )

    np.testing.assert_allclose(
        radial_mapping,
        result.internal_zeta,
        rtol=1.0e-8,
        atol=1.0e-10,
    )

    # The two copies of the material interface must share
    # the same geometrical radius.
    duplicated = np.flatnonzero(
        np.diff(result.internal_zeta) == 0.0
    )

    assert duplicated.size == 1

    lower = duplicated[0]
    upper = lower + 1

    np.testing.assert_allclose(
        result.mapping[lower],
        result.mapping[upper],
        rtol=0.0,
        atol=1.0e-12,
    )

    # The density discontinuity must be preserved.
    np.testing.assert_allclose(
        result.density[upper],
        0.4 * result.density[lower],
        rtol=1.0e-10,
        atol=0.0,
    )

    # Pressure remains continuous across the interface.
    np.testing.assert_allclose(
        result.pressure[upper],
        result.pressure[lower],
        rtol=1.0e-10,
        atol=0.0,
    )

    # The external mapping must also remain spherical.
    np.testing.assert_allclose(
        np.ptp(result.full_mapping, axis=1),
        0.0,
        rtol=0.0,
        atol=1.0e-11,
    )

    # Only the gravitational monopole may remain.
    phi_l = result.gravitational_potential_harmonics

    monopole_scale = np.max(
        np.abs(phi_l[:, 0])
    )
    nonspherical_scale = np.max(
        np.abs(phi_l[:, 1:])
    )

    assert (
        nonspherical_scale
        <= 1.0e-11 * monopole_scale
    )

    np.testing.assert_allclose(
        result.polar_radius_history[-1],
        1.0,
        rtol=0.0,
        atol=1.0e-10,
    )
    
    
def test_uniform_rotation_deforms_composite_model():
    resolution = 65
    external_resolution = 21
    angular_resolution = 9
    max_degree = 9
    mapping_precision = 1.0e-10
    density_jump = 0.4

    model = CompositePolytropeConfig(
        indices=(1.0, 1.0),
        target_pressures=(-1.0, -np.inf),
        density_jumps=(density_jump,),
        radius=1.0,
        mass=1.0,
        n_points=resolution,
    )

    result = spheroidal_method(
        model,
        solid,
        0.3,
        0.0,
        1.0,
        max_degree,
        angular_resolution,
        1,
        mapping_precision,
        3,
        2,
        OutputOptions(),
        external_resolution,
        True,
    )

    equator = np.argmin(
        np.abs(result.cos_theta)
    )
    surface = result.mapping[-1]

    np.testing.assert_allclose(
        result.cos_theta[equator],
        0.0,
        rtol=0.0,
        atol=1.0e-15,
    )

    # Equatorial symmetry is preserved.
    np.testing.assert_allclose(
        result.mapping,
        result.mapping[:, ::-1],
        rtol=0.0,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        result.full_mapping,
        result.full_mapping[:, ::-1],
        rtol=0.0,
        atol=1.0e-13,
    )

    # The equatorial radius is normalised to unity,
    # while the model is oblate.
    np.testing.assert_allclose(
        surface[equator],
        1.0,
        rtol=0.0,
        atol=1.0e-10,
    )

    assert result.polar_radius_history[-1] < 1.0

    # Material surfaces remain nested. Equality is expected
    # between the two copies of an interface.
    radial_increments = np.diff(
        result.mapping,
        axis=0,
    )

    assert radial_increments.min() >= -1.0e-12

    # Locate the duplicated material interface.
    duplicated = np.flatnonzero(
        np.diff(result.internal_zeta) == 0.0
    )

    assert duplicated.size == 1

    lower = duplicated[0]
    upper = lower + 1

    # Both material states occupy the same geometrical surface.
    np.testing.assert_allclose(
        result.mapping[lower],
        result.mapping[upper],
        rtol=0.0,
        atol=1.0e-12,
    )

    # The density jump and pressure continuity are preserved.
    np.testing.assert_allclose(
        result.density[upper],
        density_jump * result.density[lower],
        rtol=1.0e-10,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.pressure[upper],
        result.pressure[lower],
        rtol=1.0e-10,
        atol=0.0,
    )

    # The interior part of the complete mapping agrees with
    # the material mapping returned separately.
    np.testing.assert_allclose(
        result.full_mapping[result.internal_mask],
        result.mapping,
        rtol=0.0,
        atol=0.0,
    )

    # The outer boundary of the vacuum domain is spherical
    # and located at r = 2 by construction.
    np.testing.assert_allclose(
        result.full_mapping[-1],
        2.0,
        rtol=0.0,
        atol=1.0e-13,
    )

    phi_l = (
        result.gravitational_potential_harmonics
    )

    monopole_scale = np.max(
        np.abs(phi_l[:, 0])
    )
    quadrupole_scale = np.max(
        np.abs(phi_l[:, 2])
    )
    odd_scale = np.max(
        np.abs(phi_l[:, 1::2])
    )

    assert (
        quadrupole_scale
        > 1.0e-8 * monopole_scale
    )
    assert (
        odd_scale
        <= 1.0e-14 * monopole_scale
    )

    # The actual stopping criterion is satisfied.
    assert (
        abs(
            result.polar_radius_history[-1]
            - result.polar_radius_history[-2]
        )
        <= mapping_precision
    )
    
    
    reference_path = (
        Path(__file__).parent
        / "reference"
        / "spheroidal_composite_solid_omega_0p3_legacy.npz"
    )

    with np.load(reference_path) as reference:
        np.testing.assert_array_equal(
            result.zeta,
            reference["zeta"],
        )
        np.testing.assert_array_equal(
            result.internal_zeta,
            reference["internal_zeta"],
        )
        np.testing.assert_array_equal(
            result.external_zeta,
            reference["external_zeta"],
        )
        np.testing.assert_array_equal(
            result.cos_theta,
            reference["cos_theta"],
        )
        np.testing.assert_array_equal(
            result.internal_mask,
            reference["internal_mask"],
        )
        np.testing.assert_array_equal(
            result.external_mask,
            reference["external_mask"],
        )

        np.testing.assert_allclose(
            result.mapping,
            reference["mapping"],
            rtol=1.0e-9,
            atol=1.0e-11,
        )
        np.testing.assert_allclose(
            result.full_mapping,
            reference["full_mapping"],
            rtol=1.0e-9,
            atol=1.0e-11,
        )
        np.testing.assert_allclose(
            result.density,
            reference["density"],
            rtol=1.0e-9,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            result.pressure,
            reference["pressure"],
            rtol=1.0e-9,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            result.effective_potential,
            reference["effective_potential"],
            rtol=1.0e-9,
            atol=1.0e-11,
        )
        np.testing.assert_allclose(
            result.effective_potential_derivative,
            reference[
                "effective_potential_derivative"
            ],
            rtol=1.0e-9,
            atol=1.0e-11,
        )
        np.testing.assert_allclose(
            result.gravitational_potential_harmonics,
            reference[
                "gravitational_potential_harmonics"
            ],
            rtol=1.0e-9,
            atol=1.0e-11,
        )
        np.testing.assert_allclose(
            result.gravitational_potential_derivative_harmonics,
            reference[
                "gravitational_potential_derivative_harmonics"
            ],
            rtol=1.0e-9,
            atol=1.0e-11,
        )

        np.testing.assert_allclose(
            result.polar_radius_history[-1],
            reference["polar_radius"],
            rtol=1.0e-10,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            result.mass,
            reference["mass"],
            rtol=1.0e-10,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            result.radius,
            reference["radius"],
            rtol=1.0e-10,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            result.rotation_rate,
            reference["rotation_rate"],
            rtol=1.0e-10,
            atol=1.0e-12,
        )