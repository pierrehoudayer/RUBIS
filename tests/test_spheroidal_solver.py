from pathlib import Path

import numpy as np

from rubis.config import (
    CompositePolytropeConfig,
    DiagnosticOptions,
    OutputOptions,
    RotationConfig,
    SolverOptions,
)
from rubis.initialization import initialize_model_1d
from rubis.legendre import pl_project_2D
from rubis.models import Model2D, VacuumModel2D
from rubis.results import SolverInfo
from rubis.rotation_profiles import solid
from rubis.solvers import solve_spheroidal


def test_spheroidal_solver_returns_normalised_state():
    resolution = 65
    external_resolution = 21
    angular_resolution = 9
    max_degree = 9

    model = initialize_model_1d(
        CompositePolytropeConfig(
            indices=(1.0, 1.0),
            target_pressures=(-1.0, -np.inf),
            density_jumps=(0.4,),
            radius=1.0,
            mass=1.0,
            n_points=resolution,
        )
    )

    model2d, vacuum, info = solve_spheroidal(
        model,
        RotationConfig(
            profile=solid,
            target=0.0,
        ),
        SolverOptions(
            method="spheroidal",
            max_degree=max_degree,
            angular_resolution=angular_resolution,
            full_rate=1,
            mapping_precision=1.0e-10,
            spline_order=3,
            lagrange_order=2,
            external_domain_res=external_resolution,
            rescale_ab=True,
        ),
        OutputOptions(),
    )

    assert isinstance(model2d, Model2D)
    assert isinstance(vacuum, VacuumModel2D)
    assert isinstance(info, SolverInfo)
    assert info.method == "spheroidal"

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

    assert vacuum.zeta.shape == (
        external_resolution,
    )
    assert vacuum.t.shape == (
        angular_resolution,
    )
    assert vacuum.r2d.shape == (
        external_resolution,
        angular_resolution,
    )

    for field in (
        model2d.phi_g,
        model2d.phi_g_z,
        model2d.phi_c,
        model2d.phi_c_z,
        model2d.omega,
    ):
        assert field.shape == model2d.r2d.shape
        assert np.isfinite(field).all()

    for field in (
        vacuum.phi_g,
        vacuum.phi_g_z,
        vacuum.phi_c,
        vacuum.phi_c_z,
        vacuum.phi_eff,
        vacuum.phi_eff_z,
        vacuum.omega,
    ):
        assert field.shape == vacuum.r2d.shape
        assert np.isfinite(field).all()

    full_mapping = np.vstack((
        model2d.r2d,
        vacuum.r2d,
    ))
    full_zeta = np.hstack((
        model2d.zeta,
        vacuum.zeta,
    ))
    phi_g_full = np.vstack((
        model2d.phi_g,
        vacuum.phi_g,
    ))

    phi_g_l = pl_project_2D(
        phi_g_full,
        max_degree,
    )

    assert full_mapping.shape == (
        resolution + external_resolution,
        angular_resolution,
    )
    assert full_zeta.shape == (
        resolution + external_resolution,
    )
    assert phi_g_l.shape == (
        resolution + external_resolution,
        max_degree,
    )

    assert np.isfinite(model2d.r2d).all()
    assert np.isfinite(vacuum.r2d).all()
    assert np.isfinite(model2d.rho).all()
    assert np.isfinite(model2d.p).all()

    assert info.polar_radius_history.ndim == 1
    assert info.polar_radius_history.shape == (
        info.iterations + 2,
    )
    assert info.iterations >= 1
    assert info.error <= info.tolerance
    
    
def test_nonrotating_composite_model_remains_spherical():
    resolution = 65
    external_resolution = 21
    angular_resolution = 9
    max_degree = 9

    model = initialize_model_1d(
        CompositePolytropeConfig(
            indices=(1.0, 1.0),
            target_pressures=(-1.0, -np.inf),
            density_jumps=(0.4,),
            radius=1.0,
            mass=1.0,
            n_points=resolution,
        )
    )

    model2d, vacuum, info = solve_spheroidal(
        model,
        RotationConfig(
            profile=solid,
            target=0.0,
        ),
        SolverOptions(
            method="spheroidal",
            max_degree=max_degree,
            angular_resolution=angular_resolution,
            full_rate=1,
            mapping_precision=1.0e-10,
            spline_order=3,
            lagrange_order=2,
            external_domain_res=external_resolution,
            rescale_ab=True,
        ),
        OutputOptions(),
    )

    # Every material surface is spherical
    np.testing.assert_allclose(
        np.ptp(model2d.r2d, axis=1),
        0.0,
        rtol=0.0,
        atol=1.0e-11,
    )

    radial_mapping = np.mean(
        model2d.r2d,
        axis=1,
    )
    np.testing.assert_allclose(
        radial_mapping,
        model2d.zeta,
        rtol=1.0e-8,
        atol=1.0e-10,
    )

    # The two copies of the material interface share
    # the same geometrical radius
    duplicated = np.flatnonzero(
        np.diff(model2d.zeta) == 0.0
    )
    assert duplicated.size == 1

    lower = duplicated[0]
    upper = lower + 1

    np.testing.assert_allclose(
        model2d.r2d[lower],
        model2d.r2d[upper],
        rtol=0.0,
        atol=1.0e-12,
    )

    # The density discontinuity is preserved
    np.testing.assert_allclose(
        model2d.rho[upper],
        0.4 * model2d.rho[lower],
        rtol=1.0e-10,
        atol=0.0,
    )

    # Pressure remains continuous across the interface
    np.testing.assert_allclose(
        model2d.p[upper],
        model2d.p[lower],
        rtol=1.0e-10,
        atol=0.0,
    )

    # The exterior mapping remains spherical
    full_mapping = np.vstack((
        model2d.r2d,
        vacuum.r2d,
    ))
    np.testing.assert_allclose(
        np.ptp(full_mapping, axis=1),
        0.0,
        rtol=0.0,
        atol=1.0e-11,
    )

    # Only the gravitational monopole may remain
    phi_g_full = np.vstack((
        model2d.phi_g,
        vacuum.phi_g,
    ))
    phi_g_l = pl_project_2D(
        phi_g_full,
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

    np.testing.assert_allclose(
        info.polar_radius_history[-1],
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

    model = initialize_model_1d(
        CompositePolytropeConfig(
            indices=(1.0, 1.0),
            target_pressures=(-1.0, -np.inf),
            density_jumps=(density_jump,),
            radius=1.0,
            mass=1.0,
            n_points=resolution,
        )
    )

    model2d, vacuum, info = solve_spheroidal(
        model,
        RotationConfig(
            profile=solid,
            target=0.3,
        ),
        SolverOptions(
            method="spheroidal",
            max_degree=max_degree,
            angular_resolution=angular_resolution,
            full_rate=1,
            mapping_precision=mapping_precision,
            spline_order=3,
            lagrange_order=2,
            external_domain_res=external_resolution,
            rescale_ab=True,
        ),
        OutputOptions(),
    )

    full_zeta = np.hstack((
        model2d.zeta,
        vacuum.zeta,
    ))
    full_mapping = np.vstack((
        model2d.r2d,
        vacuum.r2d,
    ))

    phi_g_full = np.vstack((
        model2d.phi_g,
        vacuum.phi_g,
    ))
    phi_g_z_full = np.vstack((
        model2d.phi_g_z,
        vacuum.phi_g_z,
    ))

    phi_g_l = pl_project_2D(
        phi_g_full,
        max_degree,
    )
    phi_g_z_l = pl_project_2D(
        phi_g_z_full,
        max_degree,
    )

    internal_mask = np.hstack((
        np.ones(model2d.n_points, dtype=bool),
        np.zeros(vacuum.n_points, dtype=bool),
    ))
    external_mask = ~internal_mask

    equator = np.argmin(
        np.abs(model2d.t)
    )
    surface = model2d.r2d[-1]

    np.testing.assert_allclose(
        model2d.t[equator],
        0.0,
        rtol=0.0,
        atol=1.0e-15,
    )

    # Equatorial symmetry is preserved
    np.testing.assert_allclose(
        model2d.r2d,
        model2d.r2d[:, ::-1],
        rtol=0.0,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        full_mapping,
        full_mapping[:, ::-1],
        rtol=0.0,
        atol=1.0e-13,
    )

    # The equatorial radius is normalised to unity
    # while the model is oblate
    np.testing.assert_allclose(
        surface[equator],
        1.0,
        rtol=0.0,
        atol=1.0e-10,
    )
    assert info.polar_radius_history[-1] < 1.0

    # Material surfaces remain nested
    # Equality is expected between interface copies
    radial_increments = np.diff(
        model2d.r2d,
        axis=0,
    )
    assert radial_increments.min() >= -1.0e-12

    # Locate the duplicated material interface
    duplicated = np.flatnonzero(
        np.diff(model2d.zeta) == 0.0
    )
    assert duplicated.size == 1

    lower = duplicated[0]
    upper = lower + 1

    # Both material states occupy the same geometrical surface
    np.testing.assert_allclose(
        model2d.r2d[lower],
        model2d.r2d[upper],
        rtol=0.0,
        atol=1.0e-12,
    )

    # The density jump and pressure continuity are preserved
    np.testing.assert_allclose(
        model2d.rho[upper],
        density_jump * model2d.rho[lower],
        rtol=1.0e-10,
        atol=0.0,
    )
    np.testing.assert_allclose(
        model2d.p[upper],
        model2d.p[lower],
        rtol=1.0e-10,
        atol=0.0,
    )

    # The material part of the complete mapping agrees
    # with the material model
    np.testing.assert_allclose(
        full_mapping[internal_mask],
        model2d.r2d,
        rtol=0.0,
        atol=0.0,
    )

    # The outer vacuum boundary is spherical
    # and located at r = 2 by construction
    np.testing.assert_allclose(
        vacuum.r2d[-1],
        2.0,
        rtol=0.0,
        atol=1.0e-13,
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

    assert (
        quadrupole_scale
        > 1.0e-8 * monopole_scale
    )
    assert (
        odd_scale
        <= 1.0e-14 * monopole_scale
    )

    # The stopping criterion is satisfied
    assert info.error <= mapping_precision

    reference_path = (
        Path(__file__).parent
        / "reference"
        / "spheroidal_composite_solid_omega_0p3_legacy.npz"
    )

    with np.load(reference_path) as reference:
        reference_internal = reference["internal_mask"]

        np.testing.assert_array_equal(
            full_zeta,
            reference["zeta"],
        )
        np.testing.assert_array_equal(
            model2d.zeta,
            reference["internal_zeta"],
        )
        np.testing.assert_array_equal(
            vacuum.zeta,
            reference["external_zeta"],
        )
        np.testing.assert_array_equal(
            model2d.t,
            reference["cos_theta"],
        )
        np.testing.assert_array_equal(
            internal_mask,
            reference["internal_mask"],
        )
        np.testing.assert_array_equal(
            external_mask,
            reference["external_mask"],
        )

        np.testing.assert_allclose(
            model2d.r2d,
            reference["mapping"],
            rtol=1.0e-9,
            atol=1.0e-11,
        )
        np.testing.assert_allclose(
            full_mapping,
            reference["full_mapping"],
            rtol=1.0e-9,
            atol=1.0e-11,
        )
        np.testing.assert_allclose(
            model2d.rho,
            reference["density"],
            rtol=1.0e-9,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            model2d.p,
            reference["pressure"],
            rtol=1.0e-9,
            atol=1.0e-12,
        )

        # Model2D stores only the material effective-potential profile
        np.testing.assert_allclose(
            model2d.phi_eff,
            reference["effective_potential"][
                reference_internal
            ],
            rtol=1.0e-9,
            atol=1.0e-11,
        )
        np.testing.assert_allclose(
            model2d.phi_eff_z,
            reference[
                "effective_potential_derivative"
            ][reference_internal],
            rtol=1.0e-9,
            atol=1.0e-11,
        )

        # Harmonics are reconstructed from the physical fields
        np.testing.assert_allclose(
            phi_g_l,
            reference[
                "gravitational_potential_harmonics"
            ],
            rtol=1.0e-9,
            atol=1.0e-11,
        )
        np.testing.assert_allclose(
            phi_g_z_l,
            reference[
                "gravitational_potential_derivative_harmonics"
            ],
            rtol=1.0e-9,
            atol=1.0e-11,
        )

        np.testing.assert_allclose(
            info.polar_radius_history[-1],
            reference["polar_radius"],
            rtol=1.0e-10,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            model2d.mass,
            reference["mass"],
            rtol=1.0e-10,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            model2d.radius,
            reference["radius"],
            rtol=1.0e-10,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            model2d.omega_eq,
            reference["rotation_rate"],
            rtol=1.0e-10,
            atol=1.0e-12,
        )
        
        
def test_spheroidal_solver_runs_virial_diagnostic(
    capsys,
):
    model = initialize_model_1d(
        CompositePolytropeConfig(
            indices=(1.0, 1.0),
            target_pressures=(-1.0, -np.inf),
            density_jumps=(0.4,),
            radius=1.0,
            mass=1.0,
            n_points=65,
        )
    )

    model2d, vacuum, info = solve_spheroidal(
        model,
        RotationConfig(
            profile=solid,
            target=0.0,
        ),
        SolverOptions(
            method="spheroidal",
            max_degree=9,
            angular_resolution=9,
            full_rate=1,
            mapping_precision=1.0e-10,
            spline_order=3,
            lagrange_order=2,
            external_domain_res=21,
            rescale_ab=True,
        ),
        OutputOptions(
            diagnostics=DiagnosticOptions(
                virial_test=True,
            )
        ),
    )

    output = capsys.readouterr().out

    assert isinstance(model2d, Model2D)
    assert isinstance(vacuum, VacuumModel2D)
    assert isinstance(info, SolverInfo)
    assert "Virial theorem verified at" in output