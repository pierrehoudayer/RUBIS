import numpy as np
import pytest

from rubis.config import (
    CompositePolytropeConfig,
    PolytropeConfig,
    RotationConfig,
    SolverOptions,
)
from rubis.initialization import initialize_model_1d
from rubis.legendre import pl_project_2D
from rubis.models import Model2D
from rubis.results import SolverInfo
from rubis.rotation_profiles import solid
from rubis.solvers import solve_radial


def test_radial_solver_returns_normalised_state(capsys):
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

    model2d, vacuum, info = solve_radial(
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
    )

    assert capsys.readouterr().out == ""
    assert isinstance(model2d, Model2D)
    assert vacuum is None
    assert isinstance(info, SolverInfo)
    assert info.method == "radial"

    assert model2d.zeta.shape == (resolution,)
    assert model2d.t.shape == (angular_resolution,)
    assert model2d.r2d.shape == (
        resolution,
        angular_resolution,
    )

    assert model2d.rho.shape == (resolution,)
    assert model2d.p.shape == (resolution,)
    assert model2d.phi_eff.shape == (resolution,)
    assert model2d.phi_eff_z.shape == (resolution,)

    for field in (
        model2d.phi_g,
        model2d.phi_g_z,
        model2d.phi_c,
        model2d.phi_c_z,
        model2d.omega,
    ):
        assert field.shape == model2d.r2d.shape
        assert np.isfinite(field).all()

    phi_g_l = pl_project_2D(
        model2d.phi_g,
        max_degree,
    )
    assert phi_g_l.shape == (
        resolution,
        max_degree,
    )

    assert np.isfinite(model2d.r2d).all()
    assert np.isfinite(model2d.rho).all()
    assert np.isfinite(model2d.p).all()
    assert np.isfinite(model2d.phi_eff).all()
    assert np.isfinite(model2d.phi_eff_z).all()

    assert info.polar_radius_history.shape == (
        info.iterations + 1,
    )
    assert info.iterations >= 1
    assert info.error <= info.tolerance
    
    
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
        solve_radial(
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

    model2d, vacuum, info = solve_radial(
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
    )

    assert vacuum is None

    # Every material surface has the same radius
    # in all angular directions
    angular_spread = np.ptp(
        model2d.r2d,
        axis=1,
    )
    np.testing.assert_allclose(
        angular_spread,
        0.0,
        rtol=0.0,
        atol=1.0e-11,
    )

    # The common radius coincides with the original
    # spherical material coordinate
    radial_mapping = np.mean(
        model2d.r2d,
        axis=1,
    )
    central = model2d.zeta < 0.1

    # Outside the specially treated central region, the original
    # spherical mapping is recovered very accurately
    np.testing.assert_allclose(
        radial_mapping[~central],
        model2d.zeta[~central],
        rtol=1.0e-9,
        atol=1.0e-11,
    )

    # The central reciprocal interpolation introduces a small
    # absolute error because Phi - Phi(0) scales as r**2
    assert np.max(
        np.abs(
            radial_mapping[central]
            - model2d.zeta[central]
        )
    ) < 3.0e-6

    # Only the monopole gravitational potential may remain
    phi_g_l = pl_project_2D(
        model2d.phi_g,
        max_degree,
    )

    monopole_scale = np.max(
        np.abs(phi_g_l[:, 0])
    )
    nonspherical_scale = np.max(
        np.abs(phi_g_l[:, 1:])
    )

    assert (
        nonspherical_scale
        <= 1.0e-11 * monopole_scale
    )

    # Req is normalised to unity by construction
    np.testing.assert_allclose(
        model2d.r2d[-1],
        np.ones(angular_resolution),
        rtol=0.0,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        info.polar_radius_history[-1],
        1.0,
        rtol=0.0,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        model2d.omega_eq,
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

    model2d, vacuum, info = solve_radial(
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
            mapping_precision=mapping_precision,
            spline_order=3,
            lagrange_order=2,
            external_domain_res=21,
            rescale_ab=True,
        ),
    )

    assert vacuum is None

    equator = np.argmin(
        np.abs(model2d.t)
    )
    surface = model2d.r2d[-1]

    # With an odd Gauss-Legendre resolution, cos(theta)=0
    # is one of the angular nodes
    np.testing.assert_allclose(
        model2d.t[equator],
        0.0,
        rtol=0.0,
        atol=1.0e-15,
    )

    # The mapping remains symmetric with respect to
    # the equatorial plane
    np.testing.assert_allclose(
        model2d.r2d,
        model2d.r2d[:, ::-1],
        rtol=0.0,
        atol=1.0e-14,
    )

    # The equatorial radius is normalised to unity
    np.testing.assert_allclose(
        surface[equator],
        1.0,
        rtol=0.0,
        atol=1.0e-10,
    )

    # Uniform rotation produces an oblate surface
    assert info.polar_radius_history[-1] < 1.0
    assert surface[equator] > surface[-1]

    # Material surfaces remain nested
    radial_increments = np.diff(
        model2d.r2d,
        axis=0,
    )
    assert np.all(radial_increments > 0.0)

    # The stopping criterion is satisfied
    assert info.error <= mapping_precision

    phi_g_l = pl_project_2D(
        model2d.phi_g,
        max_degree,
    )

    monopole_scale = np.max(
        np.abs(phi_g_l[:, 0])
    )
    quadrupole_scale = np.max(
        np.abs(phi_g_l[:, 2])
    )
    odd_scale = np.max(
        np.abs(phi_g_l[:, 1::2])
    )

    # Rotation generates a measurable quadrupole
    assert (
        quadrupole_scale
        > 1.0e-8 * monopole_scale
    )

    # Equatorial symmetry suppresses odd harmonics
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
        match=r"did not converge after 1 iteration\b",    
    ):
        solve_radial(
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
        )