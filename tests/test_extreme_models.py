import numpy as np
import pytest

from rubis.config import (
    DeformationConfig,
    LegacyModelConfig,
    OutputOptions,
    PolytropeConfig, 
    RotationConfig, 
    SolverOptions,
)
from rubis.rotation_profiles import lorentzian, solid
from rubis.api import deform

@pytest.mark.slow
@pytest.mark.readme
def test_readme_near_critical_n3_model_converges():
    mapping_precision = 1.0e-10
    rotation_target = 0.9999

    model = PolytropeConfig(
        index=3.0,
        radius=1.0,
        mass=1.0,
        n_points=1001,
    )

    model2d, vacuum, info = deform(
        DeformationConfig(
            model=model,
            rotation=RotationConfig(
                profile=solid,
                target=rotation_target,
            ),
            solver=SolverOptions(
                method="radial",
                max_degree=401,
                angular_resolution=401,
                full_rate=3,
                mapping_precision=mapping_precision,
                spline_order=5,
                lagrange_order=3,
                external_domain_res=21,
                rescale_ab=True,
                max_iterations=150,
            ),
            output=OutputOptions(),
        )
    )

    assert vacuum is None
    assert info.iterations < 150

    assert np.isfinite(model2d.r2d).all()
    assert np.isfinite(model2d.phi_g).all()
    assert np.isfinite(model2d.phi_g_z).all()

    # Equatorial symmetry
    np.testing.assert_allclose(
        model2d.r2d,
        model2d.r2d[:, ::-1],
        rtol=0.0,
        atol=1.0e-12,
    )

    equator = np.argmin(
        np.abs(model2d.t)
    )

    np.testing.assert_allclose(
        model2d.t[equator],
        0.0,
        rtol=0.0,
        atol=1.0e-15,
    )

    np.testing.assert_allclose(
        model2d.r2d[-1, equator],
        1.0,
        rtol=0.0,
        atol=1.0e-10,
    )

    polar_radius = (
        info.polar_radius_history[-1]
    )

    # Broad physical bounds rather than a stored reference
    assert 0.60 < polar_radius < 0.75

    # Material surfaces remain nested
    assert np.min(
        np.diff(model2d.r2d, axis=0)
    ) >= -1.0e-11

    # The actual historical stopping criterion is met
    assert info.error <= mapping_precision

    np.testing.assert_allclose(
        model2d.omega_eq,
        rotation_target,
        rtol=1.0e-8,
        atol=1.0e-10,
    )
    
    
@pytest.mark.slow
@pytest.mark.readme
def test_readme_super_keplerian_n05_model_converges():
    mapping_precision = 1.0e-10
    rotation_target = 1.105

    model = PolytropeConfig(
        index=0.5,
        radius=1.0,
        mass=1.0,
        n_points=3001,
    )

    model2d, vacuum, info = deform(
        DeformationConfig(
            model=model,
            rotation=RotationConfig(
                profile=solid,
                target=rotation_target,
            ),
            solver=SolverOptions(
                method="radial",
                max_degree=401,
                angular_resolution=401,
                full_rate=3,
                mapping_precision=mapping_precision,
                spline_order=5,
                lagrange_order=2,
                external_domain_res=21,
                rescale_ab=True,
                max_iterations=150,
            ),
            output=OutputOptions(),
        )
    )

    assert vacuum is None
    assert info.iterations < 150

    assert np.isfinite(model2d.r2d).all()
    assert np.isfinite(model2d.rho).all()
    assert np.isfinite(model2d.p).all()
    assert np.isfinite(model2d.phi_g).all()
    assert np.isfinite(model2d.phi_g_z).all()

    # Equatorial symmetry
    np.testing.assert_allclose(
        model2d.r2d,
        model2d.r2d[:, ::-1],
        rtol=0.0,
        atol=1.0e-12,
    )

    equator = np.argmin(
        np.abs(model2d.t)
    )

    np.testing.assert_allclose(
        model2d.t[equator],
        0.0,
        rtol=0.0,
        atol=1.0e-15,
    )

    # The equatorial radius remains the normalisation radius
    np.testing.assert_allclose(
        model2d.r2d[-1, equator],
        1.0,
        rtol=0.0,
        atol=1.0e-10,
    )

    polar_radius = (
        info.polar_radius_history[-1]
    )

    # This model must be considerably more deformed than
    # the near-critical n=3 model
    assert 0.43 < polar_radius < 0.46

    # The model remains geometrically admissible
    radial_increments = np.diff(
        model2d.r2d,
        axis=0,
    )

    assert radial_increments.min() >= -1.0e-10

    # The historical stopping criterion is satisfied
    assert info.error <= mapping_precision

    # The adaptive rate must have reached the requested
    # super-Keplerian value
    np.testing.assert_allclose(
        model2d.omega_eq,
        rotation_target,
        rtol=1.0e-8,
        atol=1.0e-10,
    )
    
    
@pytest.mark.slow
@pytest.mark.readme
def test_readme_extreme_lorentzian_model_converges():
    mapping_precision = 1.0e-10
    rotation_target = 0.97
    central_diff_rate = 5.0

    model = PolytropeConfig(
        index=3.0,
        radius=1.0,
        mass=1.0,
        n_points=1001,
    )

    model2d, vacuum, info = deform(
        DeformationConfig(
            model=model,
            rotation=RotationConfig(
                profile=lorentzian,
                target=rotation_target,
                central_diff_rate=central_diff_rate,
            ),
            solver=SolverOptions(
                method="radial",
                max_degree=401,
                angular_resolution=401,
                full_rate=3,
                mapping_precision=mapping_precision,
                spline_order=5,
                lagrange_order=3,
                external_domain_res=21,
                rescale_ab=True,
                max_iterations=150,
            ),
            output=OutputOptions(),
        )
    )

    assert vacuum is None
    assert info.iterations < 150

    assert np.isfinite(model2d.r2d).all()
    assert np.isfinite(model2d.rho).all()
    assert np.isfinite(model2d.p).all()
    assert np.isfinite(model2d.phi_g).all()
    assert np.isfinite(model2d.phi_g_z).all()
    assert np.isfinite(model2d.omega).all()

    # Equatorial symmetry
    np.testing.assert_allclose(
        model2d.r2d,
        model2d.r2d[:, ::-1],
        rtol=0.0,
        atol=1.0e-12,
    )

    equator = np.argmin(
        np.abs(model2d.t)
    )

    np.testing.assert_allclose(
        model2d.t[equator],
        0.0,
        rtol=0.0,
        atol=1.0e-15,
    )

    np.testing.assert_allclose(
        model2d.r2d[-1, equator],
        1.0,
        rtol=0.0,
        atol=1.0e-10,
    )

    polar_radius = (
        info.polar_radius_history[-1]
    )

    # Broad acceptance interval around the documented model
    assert 0.24 < polar_radius < 0.28

    # Material surfaces remain nested
    radial_increments = np.diff(
        model2d.r2d,
        axis=0,
    )

    assert radial_increments.min() >= -1.0e-10

    # The historical stopping criterion is satisfied
    assert info.error <= mapping_precision

    # The adaptive equatorial rate reaches the requested value
    np.testing.assert_allclose(
        model2d.omega_eq,
        rotation_target,
        rtol=1.0e-8,
        atol=1.0e-10,
    )
    
    
@pytest.mark.slow
@pytest.mark.readme
def test_readme_jupiter_model_converges():
    mapping_precision = 1.0e-10
    rotation_target = 0.9

    model=LegacyModelConfig(
        filename="Jupiter.txt",
    )
    
    model2d, vacuum, info = deform(
        DeformationConfig(
            model=model,
            rotation=RotationConfig(
                profile=solid,
                target=rotation_target,
            ),
            solver=SolverOptions(
                method="spheroidal",
                max_degree=101,
                angular_resolution=101,
                full_rate=1,
                mapping_precision=mapping_precision,
                spline_order=5,
                lagrange_order=2,
                external_domain_res=201,
                rescale_ab=True,
                max_iterations=100,
            ),
            output=OutputOptions(),
        )
    )

    assert vacuum is not None
    assert info.iterations < 100

    full_mapping = np.vstack((
        model2d.r2d,
        vacuum.r2d,
    ))

    assert np.isfinite(model2d.r2d).all()
    assert np.isfinite(vacuum.r2d).all()
    assert np.isfinite(full_mapping).all()

    assert np.isfinite(model2d.rho).all()
    assert np.isfinite(model2d.p).all()

    assert np.isfinite(model2d.phi_g).all()
    assert np.isfinite(model2d.phi_g_z).all()
    assert np.isfinite(vacuum.phi_g).all()
    assert np.isfinite(vacuum.phi_g_z).all()

    equator = np.argmin(
        np.abs(model2d.t)
    )

    np.testing.assert_allclose(
        model2d.t[equator],
        0.0,
        rtol=0.0,
        atol=1.0e-15,
    )

    # Equatorial symmetry
    np.testing.assert_allclose(
        model2d.r2d,
        model2d.r2d[:, ::-1],
        rtol=0.0,
        atol=1.0e-12,
    )

    np.testing.assert_allclose(
        full_mapping,
        full_mapping[:, ::-1],
        rtol=0.0,
        atol=1.0e-12,
    )

    # Equatorial normalisation and strong oblateness
    np.testing.assert_allclose(
        model2d.r2d[-1, equator],
        1.0,
        rtol=0.0,
        atol=1.0e-10,
    )

    polar_radius = (
        info.polar_radius_history[-1]
    )

    assert 0.60 < polar_radius < 0.63

    # Jupiter.txt contains two duplicated material interfaces
    duplicated = np.flatnonzero(
        np.diff(model2d.zeta) == 0.0
    )

    assert duplicated.size == 2

    for lower in duplicated:
        upper = lower + 1

        # Both sides occupy the same geometrical interface
        np.testing.assert_allclose(
            model2d.r2d[lower],
            model2d.r2d[upper],
            rtol=0.0,
            atol=1.0e-11,
        )

        # Pressure remains continuous
        np.testing.assert_allclose(
            model2d.p[lower],
            model2d.p[upper],
            rtol=1.0e-9,
            atol=1.0e-13,
        )

        # The interface represents a genuine density jump
        assert not np.isclose(
            model2d.rho[lower],
            model2d.rho[upper],
            rtol=1.0e-3,
            atol=0.0,
        )

    # Surfaces remain ordered; zero increments are legitimate
    # at duplicated interfaces
    radial_increments = np.diff(
        model2d.r2d,
        axis=0,
    )

    assert radial_increments.min() >= -1.0e-10

    # The numerical vacuum domain returns to a spherical
    # outer boundary at r = 2
    np.testing.assert_allclose(
        vacuum.r2d[-1],
        2.0,
        rtol=0.0,
        atol=1.0e-12,
    )

    # The historical convergence criterion is met
    assert info.error <= mapping_precision

    np.testing.assert_allclose(
        model2d.omega_eq,
        rotation_target,
        rtol=1.0e-8,
        atol=1.0e-10,
    )