import numpy as np

from rubis.config import RadiativeFluxOptions
from rubis.domains import find_domains
from rubis.flux import (
    RadiativeFlux,
    compute_radiative_flux,
)
from rubis.mapping import initialize_mapping
from rubis.models import Model2D


def make_spherical_model():
    zeta = np.linspace(
        0.0,
        1.0,
        41,
    )
    r2d, t = initialize_mapping(zeta, M=21)

    zeros_1d = np.zeros_like(zeta)
    zeros_2d = np.zeros_like(r2d)

    return Model2D(
        G=1.0,
        surface_pressure=0.0,
        mass=1.0,
        radius=1.0,
        omega_eq=0.0,
        zeta=zeta,
        t=t,
        r2d=r2d,
        rho=np.ones_like(zeta),
        p=np.ones_like(zeta),
        additional_variables=(),
        phi_eff=zeros_1d,
        phi_eff_z=zeros_1d,
        phi_g=zeros_2d,
        phi_g_z=zeros_2d,
        phi_c=zeros_2d,
        phi_c_z=zeros_2d,
        omega=zeros_2d,
        domains=find_domains(zeta),
    )


def test_radiative_flux_is_uniform_for_spherical_model():
    model = make_spherical_model()

    flux = compute_radiative_flux(
        model,
        RadiativeFluxOptions(
            origin=0.2,
            n_lines=8,
            max_degree=11,
            spline_order=3,
        ),
    )

    assert isinstance(
        flux,
        RadiativeFlux,
    )

    assert flux.line_zeta.ndim == 1
    assert flux.line_r.shape == (
        flux.line_zeta.size,
        16,
    )
    assert flux.line_t.shape == (
        flux.line_zeta.size,
        16,
    )

    assert flux.surface_t.shape == (16,)
    assert flux.surface_flux.shape == (16,)
    assert flux.surface_flux_l.shape == (16,)

    assert np.isfinite(
        flux.line_r
    ).all()
    assert np.isfinite(
        flux.line_t
    ).all()
    assert np.isfinite(
        flux.surface_flux
    ).all()
    assert np.isfinite(
        flux.surface_flux_l
    ).all()

    assert flux.surface_flux_l[0] > 0.0

    assert (
        np.linalg.norm(
            flux.surface_flux_l[1:]
        )
        < 1.0e-6
        * abs(flux.surface_flux_l[0])
    )

    np.testing.assert_allclose(
        flux.surface_flux,
        np.mean(flux.surface_flux),
        rtol=1.0e-6,
        atol=1.0e-8,
    )