import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pytest

from rubis.domains import find_domains
from rubis.flux import RadiativeFlux
from rubis.models import Model2D, VacuumModel2D
from rubis.plotting import (
    STELLAR_CMAP,
    plot_gravitational_harmonics,
    plot_model_field,
    plot_radiative_flux_lines,
    plot_radiative_flux_surface,
)


def make_spherical_model():
    zeta = np.linspace(0.0, 1.0, 21)
    t = np.polynomial.legendre.leggauss(9)[0]
    r2d = np.repeat(zeta[:, None], t.size, axis=1)

    rho = 1.0 - 0.5 * zeta**2
    phi_eff = zeta**2
    phi_g = np.repeat(
        (-1.0 + 0.5 * zeta)[:, None],
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
        omega_eq=0.0,
        zeta=zeta,
        t=t,
        r2d=r2d,
        rho=rho,
        p=rho,
        additional_variables=(),
        phi_eff=phi_eff,
        phi_eff_z=zeros_1d,
        phi_g=phi_g,
        phi_g_z=zeros_2d,
        phi_c=zeros_2d,
        phi_c_z=zeros_2d,
        omega=zeros_2d,
        domains=find_domains(zeta),
    )


def make_vacuum(model):
    zeta = np.linspace(1.0, 1.5, 11)
    r2d = np.repeat(
        zeta[:, None],
        model.angular_resolution,
        axis=1,
    )
    phi_g = -1.0 / r2d
    zeros = np.zeros_like(r2d)

    return VacuumModel2D(
        G=model.G,
        mass=model.mass,
        radius=model.radius,
        omega_eq=model.omega_eq,
        zeta=zeta,
        t=model.t.copy(),
        r2d=r2d,
        phi_g=phi_g,
        phi_g_z=zeros,
        phi_c=zeros,
        phi_c_z=zeros,
        phi_eff=phi_g.copy(),
        phi_eff_z=zeros,
        omega=zeros,
        domains=find_domains(zeta),
    )


def make_flux():
    line_zeta = np.linspace(0.2, 1.0, 5)
    line_t = np.tile(
        np.array([-0.5, 0.5]),
        (line_zeta.size, 1),
    )
    line_r = np.repeat(line_zeta[:, None], 2, axis=1)
    surface_t = np.linspace(-1.0, 1.0, 9)
    surface_flux = np.ones(surface_t.size)
    surface_flux_l = np.zeros(surface_t.size)
    surface_flux_l[0] = 1.0

    return RadiativeFlux(
        line_zeta=line_zeta,
        line_r=line_r,
        line_t=line_t,
        surface_t=surface_t,
        surface_flux=surface_flux,
        surface_flux_l=surface_flux_l,
        max_degree=9,
    )


def test_stellar_colormap_is_matplotlib_colormap():
    assert isinstance(STELLAR_CMAP, matplotlib.colors.Colormap)


def test_model_field_plot_returns_figure():
    model = make_spherical_model()
    fig, ax = plot_model_field(
        model,
        model.rho,
        angular_resolution=51,
    )

    assert fig is ax.figure
    assert len(ax.collections) > 0

    plt.close(fig)


def test_model_field_plot_accepts_vacuum_and_surfaces():
    model = make_spherical_model()
    vacuum = make_vacuum(model)
    fig, ax = plot_model_field(
        model,
        model.phi_g,
        vacuum=vacuum,
        angular_resolution=51,
        show_surfaces=True,
        surface_count=5,
        vacuum_surface_count=5,
    )

    assert fig is ax.figure
    assert len(ax.lines) > 0

    plt.close(fig)


def test_model_field_rejects_angular_derivative_of_profile():
    model = make_spherical_model()

    with pytest.raises(ValueError, match="no angular derivative"):
        plot_model_field(
            model,
            model.rho,
            angular_derivative=1,
        )


def test_gravitational_harmonics_returns_figure():
    model = make_spherical_model()
    vacuum = make_vacuum(model)
    fig, ax = plot_gravitational_harmonics(
        model,
        vacuum=vacuum,
    )

    assert fig is ax.figure
    assert len(ax.lines) > 0

    plt.close(fig)


def test_flux_line_plot_returns_figure():
    flux = make_flux()
    fig, ax = plot_radiative_flux_lines(
        flux,
        color="0.3",
    )

    assert fig is ax.figure
    assert len(ax.lines) > 0

    plt.close(fig)


def test_flux_surface_plot_returns_figure():
    model = make_spherical_model()
    flux = make_flux()
    fig, ax = plot_radiative_flux_surface(
        model,
        flux,
        resolution=(20, 20),
        cmap="stellar",
    )

    assert fig is ax.figure
    assert len(ax.collections) > 0

    plt.close(fig)


def test_plotting_does_not_modify_global_rcparams():
    flux = make_flux()
    keys = (
        "text.usetex",
        "xtick.labelsize",
        "ytick.labelsize",
        "axes.facecolor",
    )
    before = {
        key: matplotlib.rcParams[key]
        for key in keys
    }

    fig, _ = plot_radiative_flux_lines(flux)
    after = {
        key: matplotlib.rcParams[key]
        for key in keys
    }

    assert after == before

    plt.close(fig)
