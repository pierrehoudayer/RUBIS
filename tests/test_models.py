import numpy as np

from rubis.domains import find_domains
from rubis.models import Model1D


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