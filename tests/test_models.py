import numpy as np

from rubis.domains import find_domains
from rubis.models import (
    Model1D,
    Model2D,
    VacuumModel2D,
)


def test_model_1d_reports_grid_properties():
    r = np.linspace(0.0, 1.0, 5)

    model = Model1D(
        G=6.67384e-8,
        surface_pressure=0.0,
        mass=2.0,
        radius=3.0,
        r=r,
        rho=np.ones_like(r),
        domains=find_domains(r),
    )

    assert model.n_points == 5
    assert model.n_domains == 1
    assert model.additional_variables == ()


def test_model_1d_reports_multiple_domains():
    r = np.array([
        0.0,
        0.5,
        0.5,
        1.0,
    ])

    model = Model1D(
        G=6.67384e-8,
        surface_pressure=0.0,
        mass=1.0,
        radius=1.0,
        r=r,
        rho=np.ones_like(r),
        domains=find_domains(r),
    )

    assert model.n_points == 4
    assert model.n_domains == 2
    assert model.domains.has_interfaces


def test_model_1d_preserves_additional_variables():
    r = np.linspace(0.0, 1.0, 3)
    temperature = np.array([1.0, 2.0, 3.0])

    model = Model1D(
        G=6.67384e-8,
        surface_pressure=0.0,
        mass=1.0,
        radius=1.0,
        r=r,
        rho=np.ones_like(r),
        domains=find_domains(r),
        additional_variables=(temperature,),
    )

    assert len(model.additional_variables) == 1
    np.testing.assert_array_equal(
        model.additional_variables[0],
        temperature,
    )
    
    
def test_model_2d_reports_material_dimensions():
    zeta = np.linspace(0.0, 1.0, 5)
    t = np.linspace(-1.0, 1.0, 3)
    r2d = zeta[:, None] * np.ones_like(t)

    model = Model2D(
        G=1.0,
        surface_pressure=0.0,
        mass=1.0,
        radius=1.0,
        omega_eq=0.2,
        zeta=zeta,
        t=t,
        r2d=r2d,
        rho=np.ones_like(zeta),
        p=np.ones_like(zeta),
        additional_variables=(),
        phi_eff=np.zeros_like(zeta),
        phi_eff_z=np.zeros_like(zeta),
        phi_g=np.zeros_like(r2d),
        phi_g_z=np.zeros_like(r2d),
        phi_c=np.zeros_like(r2d),
        phi_c_z=np.zeros_like(r2d),
        omega=np.full_like(r2d, 0.2),
        domains=find_domains(zeta),
    )

    assert model.n_points == 5
    assert model.angular_resolution == 3
    assert model.n_domains == 1
    
    
def test_vacuum_model_2d_reports_exterior_dimensions():
    zeta = np.linspace(1.0, 2.0, 4)
    t = np.linspace(-1.0, 1.0, 3)
    r2d = zeta[:, None] * np.ones_like(t)

    vacuum = VacuumModel2D(
        G=1.0,
        mass=1.0,
        radius=1.0,
        omega_eq=0.2,
        zeta=zeta,
        t=t,
        r2d=r2d,
        phi_g=np.zeros_like(r2d),
        phi_g_z=np.zeros_like(r2d),
        phi_c=np.zeros_like(r2d),
        phi_c_z=np.zeros_like(r2d),
        phi_eff=np.zeros_like(r2d),
        phi_eff_z=np.zeros_like(r2d),
        omega=np.full_like(r2d, 0.2),
        domains=find_domains(zeta),
    )

    assert vacuum.n_points == 4
    assert vacuum.angular_resolution == 3
    assert vacuum.n_domains == 1