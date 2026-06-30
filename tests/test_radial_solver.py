import pytest

import numpy as np

from rubis.options import OutputOptions
from rubis.rotation_profiles import solid
from rubis.config import (
    CompositePolytropeConfig, 
    PolytropeConfig, 
    RotationConfig,
    SolverOptions,
)
from rubis.initialization import initialize_model_1d
from rubis.solvers import radial_method
from rubis.results import DeformationResult, RadialResult


def test_radial_solver_returns_normalised_state():
    resolution = 65
    angular_resolution = 9
    max_degree = 9

    model = initialize_model_1d(
        PolytropeConfig(
            index=1.0,
            radius=1.0,
            mass=1.0,
            n_points=resolution,
        )
    )

    result = radial_method(
        model,
        RotationConfig(
            profile=solid,
            target=0.0,
        ),
        SolverOptions(
            method="radial",
            max_degree=max_degree,
            angular_resolution=angular_resolution,
            full_rate=1,
            mapping_precision=1.0e-10,
            spline_order=3,
            lagrange_order=2,
            external_domain_res=21,
            rescale_ab=True,
        ),
        OutputOptions(),
    )

    assert isinstance(result, RadialResult)
    assert isinstance(result, DeformationResult)
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
    
    
def test_radial_solver_rejects_multidomain_model():
    model = initialize_model_1d(
        CompositePolytropeConfig(
            indices=(1.0, 1.0),
            target_pressures=(-1.0, -np.inf),
            density_jumps=(0.4,),
            n_points=65,
        )
    )

    with pytest.raises(ValueError, match="single-domain"):
        radial_method(
            model,
            RotationConfig(
                profile=solid,
                target=0.0,
            ),
            SolverOptions(
                method="radial",
                max_degree=9,
                angular_resolution=9,
                full_rate=1,
                mapping_precision=1.0e-10,
                spline_order=3,
                lagrange_order=2,
                external_domain_res=21,
                rescale_ab=True,
            ),
            OutputOptions(),
        )
    
    
def test_nonrotating_radial_model_remains_spherical():
    resolution = 65
    angular_resolution = 9
    max_degree = 9

    model = initialize_model_1d(
        PolytropeConfig(
            index=1.0,
            radius=1.0,
            mass=1.0,
            n_points=resolution,
        )
    )

    result = radial_method(
        model,
        RotationConfig(
            profile=solid,
            target=0.0,
        ),
        SolverOptions(
            method="radial",
            max_degree=max_degree,
            angular_resolution=angular_resolution,
            full_rate=1,
            mapping_precision=1.0e-10,
            spline_order=3,
            lagrange_order=2,
            external_domain_res=21,
            rescale_ab=True,
        ),
        OutputOptions(),
    )

    # Every material surface must have the same radius
    # in all angular directions.
    angular_spread = np.ptp(
        result.mapping,
        axis=1,
    )

    np.testing.assert_allclose(
        angular_spread,
        0.0,
        rtol=0.0,
        atol=1.0e-11,
    )

    # That common radius must coincide with the original
    # spherical material coordinate.
    radial_mapping = np.mean(
        result.mapping,
        axis=1,
    )

    central = result.zeta < 0.1

    # Outside the specially treated central region, the original
    # spherical mapping should be recovered very accurately.
    np.testing.assert_allclose(
        radial_mapping[~central],
        result.zeta[~central],
        rtol=1.0e-9,
        atol=1.0e-11,
    )

    # The central reciprocal interpolation introduces a small
    # absolute error because Phi - Phi(0) scales as r**2.
    assert np.max(
        np.abs(
            radial_mapping[central]
            - result.zeta[central]
        )
    ) < 3.0e-6

    # Only the monopole gravitational potential may remain.
    monopole_scale = np.max(
        np.abs(
            result.gravitational_potential_harmonics[:, 0]
        )
    )
    nonspherical_scale = np.max(
        np.abs(
            result.gravitational_potential_harmonics[:, 1:]
        )
    )

    assert nonspherical_scale <= 1.0e-11 * monopole_scale

    # Req is normalised to unity by construction.
    np.testing.assert_allclose(
        result.mapping[-1],
        np.ones(angular_resolution),
        rtol=0.0,
        atol=1.0e-10,
    )

    np.testing.assert_allclose(
        result.polar_radius_history[-1],
        1.0,
        rtol=0.0,
        atol=1.0e-10,
    )

    np.testing.assert_allclose(
        result.rotation_rate,
        0.0,
        rtol=0.0,
        atol=0.0,
    )
    
    
def test_uniform_rotation_produces_oblate_model():
    resolution = 65
    angular_resolution = 9
    max_degree = 9
    mapping_precision = 1.0e-10

    model = initialize_model_1d(
        PolytropeConfig(
            index=1.0,
            radius=1.0,
            mass=1.0,
            n_points=resolution,
        )
    )

    result = radial_method(
        model,
        RotationConfig(
            profile=solid,
            target=0.3,
        ),
        SolverOptions(
            method="radial",
            max_degree=max_degree,
            angular_resolution=angular_resolution,
            full_rate=1,
            mapping_precision=1.0e-10,
            spline_order=3,
            lagrange_order=2,
            external_domain_res=21,
            rescale_ab=True,
        ),
        OutputOptions(),
    )
    
    equator = np.argmin(
        np.abs(result.cos_theta)
    )
    surface = result.mapping[-1]

    # With an odd Gauss-Legendre resolution, cos(theta)=0
    # is one of the angular nodes.
    np.testing.assert_allclose(
        result.cos_theta[equator],
        0.0,
        rtol=0.0,
        atol=1.0e-15,
    )

    # The mapping must remain symmetric with respect to
    # the equatorial plane.
    np.testing.assert_allclose(
        result.mapping,
        result.mapping[:, ::-1],
        rtol=0.0,
        atol=1.0e-14,
    )

    # The equatorial radius is normalised to unity.
    np.testing.assert_allclose(
        surface[equator],
        1.0,
        rtol=0.0,
        atol=1.0e-10,
    )

    # Uniform rotation produces an oblate surface.
    assert result.polar_radius_history[-1] < 1.0
    assert surface[equator] > surface[-1]

    # Material surfaces must remain nested.
    radial_increments = np.diff(
        result.mapping,
        axis=0,
    )

    assert np.all(radial_increments > 0.0)

    # The stopping criterion must be satisfied.
    assert (
        abs(
            result.polar_radius_history[-1]
            - result.polar_radius_history[-2]
        )
        <= mapping_precision
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

    # Rotation generates a measurable quadrupole.
    assert (
        quadrupole_scale
        > 1.0e-8 * monopole_scale
    )

    # Equatorial symmetry suppresses odd harmonics.
    assert (
        odd_scale
        <= 1.0e-14 * monopole_scale
    )
    
    
def test_radial_solver_enforces_iteration_limit():
    resolution = 65
    angular_resolution = 9
    max_degree = 9
    mapping_precision = 1.0e-10
    
    model = initialize_model_1d(
        PolytropeConfig(
            index=1.0,
            radius=1.0,
            mass=1.0,
            n_points=resolution,
        )
    )

    with pytest.raises(
        RuntimeError,
        match="did not converge after 1 iterations",
    ):
        radial_method(
            model,
            RotationConfig(
                profile=solid,
                target=0.3,
            ),
            SolverOptions(
                method="radial",
                max_degree=max_degree,
                angular_resolution=angular_resolution,
                full_rate=1,
                mapping_precision=1.0e-10,
                spline_order=3,
                lagrange_order=2,
                external_domain_res=21,
                rescale_ab=True,
                max_iterations=1,
            ),
            OutputOptions(),
        )