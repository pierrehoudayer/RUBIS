import numpy as np

from rubis.hydrostatics import integrate_pressure


def test_integrate_pressure_recovers_quadratic_profile():
    zeta = np.linspace(0.0, 1.0, 21)
    rho = np.ones_like(zeta)
    dphi_eff = 2.0 * zeta
    surface_pressure = 0.1

    p = integrate_pressure(
        zeta,
        rho,
        dphi_eff,
        surface_pressure,
        spline_order=3,
    )

    expected = surface_pressure + 1.0 - zeta**2

    np.testing.assert_allclose(
        p,
        expected,
        atol=1.0e-12,
    )
    assert p[-1] == surface_pressure
    
    
def test_integrate_pressure_handles_duplicated_interface():
    zeta = np.array([
        0.0,
        0.25,
        0.5,
        0.5,
        0.75,
        1.0,
    ])
    unique = np.array([0, 1, 2, 4, 5])

    rho = np.ones_like(zeta)
    dphi_eff = 2.0 * zeta
    surface_pressure = 0.1

    p = integrate_pressure(
        zeta,
        rho,
        dphi_eff,
        surface_pressure,
        unique_indices=unique,
        spline_order=3,
    )

    expected = surface_pressure + 1.0 - zeta**2

    np.testing.assert_allclose(
        p,
        expected,
        atol=1.0e-12,
    )
    assert p[2] == p[3]