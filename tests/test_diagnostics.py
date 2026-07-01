import numpy as np

from rubis.diagnostics import (
    GravitationalMoments,
    VirialBalance,
    compute_gravitational_moments,
    compute_virial_balance,
    report_virial_balance,
)


def test_virial_balance_computes_relative_residual():
    balance = VirialBalance(
        kinetic_energy=1.0,
        potential_work=4.0,
        thermodynamic_work=-1.0,
        surface_work=3.0,
    )

    assert balance.residual == 0.0
    assert balance.relative_residual == 0.0


def test_report_virial_balance_displays_result(capsys):
    balance = VirialBalance(
        kinetic_energy=1.0,
        potential_work=4.0,
        thermodynamic_work=-1.0,
        surface_work=3.0,
    )

    report_virial_balance(balance)

    output = capsys.readouterr().out
    assert "Virial theorem verified at" in output
    
    
def test_gravitational_moments_are_spherical():
    r = np.linspace(0.0, 1.0, 65)
    t = np.polynomial.legendre.leggauss(15)[0]

    r2d = np.repeat(r[:, None], t.size, axis=1)
    rho = np.ones_like(r)

    moments = compute_gravitational_moments(
        r2d,
        rho,
        t,
        max_degree=6,
        spline_order=3,
    )

    assert moments.values[0] > 0.0
    np.testing.assert_allclose(
        moments.values[1:],
        0.0,
        atol=1.0e-10,
    )