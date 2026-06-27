import numpy as np

from polytrope import polytrope


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