import numpy as np

from rubis.diagnostics import (
    VirialBalance,
    compute_gravitational_moments,
    compute_virial_balance,
    report_virial_balance,
)
from rubis.domains import find_domains
from rubis.models import Model2D


def make_spherical_model(
    zeta,
    rho,
    *,
    omega=0.0,
    phi_g=0.0,
):
    t = np.polynomial.legendre.leggauss(
        15
    )[0]

    r2d = np.repeat(
        zeta[:, None],
        t.size,
        axis=1,
    )

    zeros_1d = np.zeros_like(zeta)
    zeros_2d = np.zeros_like(r2d)

    return Model2D(
        G=1.0,
        surface_pressure=0.0,
        mass=1.0,
        radius=1.0,
        omega_eq=omega,

        zeta=zeta,
        t=t,
        r2d=r2d,

        rho=rho,
        p=zeros_1d,
        additional_variables=(),

        phi_eff=zeros_1d,
        phi_eff_z=zeros_1d,

        phi_g=np.full_like(
            r2d,
            phi_g,
        ),
        phi_g_z=zeros_2d,

        phi_c=zeros_2d,
        phi_c_z=zeros_2d,

        omega=np.full_like(
            r2d,
            omega,
        ),

        domains=find_domains(zeta),
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


def test_report_virial_balance_displays_result(
    capsys,
):
    balance = VirialBalance(
        kinetic_energy=1.0,
        potential_work=4.0,
        thermodynamic_work=-1.0,
        surface_work=3.0,
    )

    report_virial_balance(balance)

    output = capsys.readouterr().out

    assert "Virial theorem verified at" in output


def test_virial_balance_uses_model_fields():
    zeta = np.linspace(
        0.0,
        1.0,
        65,
    )

    model = make_spherical_model(
        zeta,
        np.ones_like(zeta),
        omega=1.0,
        phi_g=-1.0,
    )

    balance = compute_virial_balance(
        model,
        spline_order=3,
    )

    np.testing.assert_allclose(
        balance.potential_work,
        4.0 * np.pi / 3.0,
        rtol=1.0e-10,
        atol=1.0e-12,
    )

    np.testing.assert_allclose(
        balance.kinetic_energy,
        4.0 * np.pi / 15.0,
        rtol=1.0e-10,
        atol=1.0e-12,
    )

    np.testing.assert_allclose(
        balance.thermodynamic_work,
        0.0,
        rtol=0.0,
        atol=1.0e-12,
    )

    np.testing.assert_allclose(
        balance.surface_work,
        0.0,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_gravitational_moments_use_material_domains():
    zeta = np.hstack((
        np.linspace(
            0.0,
            0.5,
            33,
        ),
        np.linspace(
            0.5,
            1.0,
            33,
        ),
    ))

    rho = np.hstack((
        np.ones(33),
        0.5 * np.ones(33),
    ))

    model = make_spherical_model(
        zeta,
        rho,
    )

    moments = compute_gravitational_moments(
        model,
        max_degree=6,
        spline_order=3,
    )

    # Integral of a spherical piecewise-constant density:
    # 4 pi [int_0^.5 r² dr + .5 int_.5^1 r² dr]
    np.testing.assert_allclose(
        moments.values[0],
        3.0 * np.pi / 4.0,
        rtol=1.0e-10,
        atol=1.0e-12,
    )

    np.testing.assert_allclose(
        moments.values[1:],
        0.0,
        rtol=0.0,
        atol=1.0e-10,
    )