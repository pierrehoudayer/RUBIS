import pytest

import numpy as np
from scipy.integrate import simpson

from rubis.models import CompositePolytropeConfig
from rubis.polytrope import composite_polytrope, polytrope


G = 6.67384e-8


def test_n1_polytrope_matches_analytic_solution():
    radius = 2.0
    mass = 3.0
    resolution = 101

    model = polytrope(
        1.0,
        R=radius,
        M=mass,
        res=resolution,
    )

    x = np.pi * model.r / radius
    enthalpy = np.sinc(x / np.pi)

    central_density = (
        np.pi * mass / (4.0 * radius**3)
    )
    central_pressure = (
        np.pi * G * mass**2 / (8.0 * radius**4)
    )

    np.testing.assert_allclose(
        model.rho,
        central_density * enthalpy,
        rtol=1.0e-14,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        model.p,
        central_pressure * enthalpy**2,
        rtol=1.0e-14,
        atol=1.0e-15,
    )

    np.testing.assert_allclose(
        model.r[[0, -1]],
        np.array([0.0, radius]),
        rtol=0.0,
        atol=1.0e-14,
    )


def test_n1_polytrope_has_regular_central_gravity():
    model = polytrope(
        1.0,
        R=2.0,
        M=3.0,
        res=101,
    )

    assert np.isfinite(model.g).all()

    np.testing.assert_allclose(
        model.g[0],
        0.0,
        rtol=0.0,
        atol=0.0,
    )

    np.testing.assert_allclose(
        model.g[-1],
        G * 3.0 / 2.0**2,
        rtol=1.0e-14,
        atol=1.0e-15,
    )
    
    
def test_n0_polytrope_matches_uniform_sphere():
    radius = 2.0
    mass = 3.0

    model = polytrope(
        0.0,
        R=radius,
        M=mass,
        res=101,
    )

    expected_density = (
        3.0 * mass
        / (4.0 * np.pi * radius**3)
    )
    expected_gravity = (
        G * mass * model.r / radius**3
    )
    expected_pressure = (
        3.0 * G * mass**2
        / (8.0 * np.pi * radius**4)
        * (1.0 - (model.r / radius)**2)
    )

    np.testing.assert_allclose(
        model.rho,
        expected_density,
        rtol=2.0e-13,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        model.g,
        expected_gravity,
        rtol=2.0e-13,
        atol=1.0e-20,
    )
    np.testing.assert_allclose(
        model.p,
        expected_pressure,
        rtol=2.0e-13,
        atol=1.0e-22,
    )

    np.testing.assert_allclose(
        model.r[[0, -1]],
        np.array([0.0, radius]),
        rtol=0.0,
        atol=1.0e-14,
    )
    
    
def test_n3_polytrope_global_properties():
    radius = 2.0
    mass = 3.0

    model = polytrope(
        3.0,
        R=radius,
        M=mass,
        res=501,
    )

    assert np.isfinite(model.r).all()
    assert np.isfinite(model.rho).all()
    assert np.isfinite(model.p).all()
    assert np.isfinite(model.g).all()

    np.testing.assert_allclose(
        model.r[[0, -1]],
        np.array([0.0, radius]),
        rtol=0.0,
        atol=1.0e-12,
    )

    np.testing.assert_allclose(
        model.g[0],
        0.0,
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        model.g[-1],
        G * mass / radius**2,
        rtol=1.0e-10,
        atol=1.0e-15,
    )

    integrated_mass = 4.0 * np.pi * simpson(
        model.rho * model.r**2,
        x=model.r,
    )

    np.testing.assert_allclose(
        integrated_mass,
        mass,
        rtol=1.0e-10,
        atol=0.0,
    )

    assert np.all(np.diff(model.r) > 0.0)
    assert np.all(np.diff(model.rho) <= 0.0)
    assert np.all(np.diff(model.p) <= 0.0)

    assert np.all(model.rho >= 0.0)
    assert np.all(model.p >= 0.0)
    assert np.all(model.g >= 0.0)
    
    
def test_single_region_composite_matches_simple_polytrope():
    radius = 2.0
    mass = 3.0
    resolution = 501
    index = 3.0

    simple_model = polytrope(
        index,
        R=radius,
        M=mass,
        res=resolution,
    )

    composite_model = composite_polytrope(
        CompositePolytropeConfig(
            indices=index,
            target_pressures=-np.inf,
            density_jumps=None,
            radius=radius,
            mass=mass,
            resolution=resolution,
        )
    )

    np.testing.assert_allclose(
        composite_model.r,
        simple_model.r,
        rtol=1.0e-13,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        composite_model.rho,
        simple_model.rho,
        rtol=1.0e-10,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        composite_model.p,
        simple_model.p,
        rtol=1.0e-10,
        atol=1.0e-22,
    )
    np.testing.assert_allclose(
        composite_model.g,
        simple_model.g,
        rtol=1.0e-10,
        atol=1.0e-20,
    )
    
    
def test_composite_polytrope_interface_conditions():
    radius = 2.0
    mass = 3.0
    resolution = 301
    density_jump = 0.4

    model = composite_polytrope(
        CompositePolytropeConfig(
            indices=(1.0, 1.0),
            target_pressures=(-1.0, -np.inf),
            density_jumps=(density_jump,),
            radius=radius,
            mass=mass,
            resolution=resolution,
        )
    )

    assert model.r.size == resolution
    assert model.rho.size == resolution
    assert model.p.size == resolution
    assert model.g.size == resolution

    # A two-region model contains one duplicated radius.
    interface_indices = np.flatnonzero(
        np.diff(model.r) == 0.0
    )

    assert interface_indices.size == 1

    lower = interface_indices[0]
    upper = lower + 1

    # The two copies represent the same geometrical surface.
    np.testing.assert_equal(
        model.r[lower],
        model.r[upper],
    )

    # Pressure and gravity remain continuous.
    np.testing.assert_allclose(
        model.p[upper],
        model.p[lower],
        rtol=1.0e-11,
        atol=0.0,
    )
    np.testing.assert_allclose(
        model.g[upper],
        model.g[lower],
        rtol=1.0e-11,
        atol=0.0,
    )

    # Density follows the prescribed jump from the inner
    # to the outer side of the interface.
    np.testing.assert_allclose(
        model.rho[upper],
        density_jump * model.rho[lower],
        rtol=1.0e-11,
        atol=0.0,
    )

    # The interface is placed at P / Pc = 10^-1.
    np.testing.assert_allclose(
        model.p[lower] / model.p[0],
        1.0e-1,
        rtol=1.0e-11,
        atol=0.0,
    )

    assert np.all(np.diff(model.r) >= 0.0)
    
    
def test_composite_polytrope_rejects_invalid_pressure_count():
    config = CompositePolytropeConfig(
        indices=(2.0, 1.0),
        target_pressures=(-np.inf,),
    )

    with pytest.raises(ValueError, match="target_pressures"):
        composite_polytrope(config)


def test_composite_polytrope_rejects_invalid_density_jump_count():
    config = CompositePolytropeConfig(
        indices=(2.0, 1.0),
        target_pressures=(-1.0, -np.inf),
        density_jumps=(0.5, 0.8),
    )

    with pytest.raises(ValueError, match="density_jumps"):
        composite_polytrope(config)