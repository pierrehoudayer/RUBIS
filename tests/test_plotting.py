import matplotlib

matplotlib.use(
    "Agg",
    force=True,
)

import matplotlib.pyplot as plt
import numpy as np

from rubis.domains import find_domains
from rubis.flux import RadiativeFlux
from rubis.mapping import initialize_mapping
from rubis.models import Model2D
from rubis.plotting import (
    STELLAR_CMAP,
    plot_radiative_flux_lines,
    plot_radiative_flux_surface,
)


def make_spherical_model():
    zeta = np.linspace(
        0.0,
        1.0,
        21,
    )

    r2d, t = initialize_mapping(
        zeta,
        M=9,
    )

    zeros_1d = np.zeros_like(
        zeta
    )
    zeros_2d = np.zeros_like(
        r2d
    )

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


def make_flux():
    line_zeta = np.linspace(
        0.2,
        1.0,
        5,
    )

    line_t = np.column_stack((
        np.full(
            line_zeta.size,
            -0.5,
        ),
        np.full(
            line_zeta.size,
            0.5,
        ),
    ))

    line_r = np.repeat(
        line_zeta[:, None],
        2,
        axis=1,
    )

    surface_t = np.linspace(
        -1.0,
        1.0,
        9,
    )

    surface_flux = np.ones(
        surface_t.size
    )

    surface_flux_l = np.zeros(
        surface_t.size
    )
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
    assert isinstance(
        STELLAR_CMAP,
        matplotlib.colors.Colormap,
    )


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
        resolution=(
            20,
            20,
        ),
        cmap="stellar",
    )

    assert fig is ax.figure
    assert len(ax.collections) > 0

    plt.close(fig)


def test_plotting_does_not_modify_global_rcparams():
    flux = make_flux()

    before = {
        key: matplotlib.rcParams[key]
        for key in (
            "text.usetex",
            "xtick.labelsize",
            "ytick.labelsize",
            "axes.facecolor",
        )
    }

    fig, _ = plot_radiative_flux_lines(
        flux
    )

    after = {
        key: matplotlib.rcParams[key]
        for key in before
    }

    assert after == before

    plt.close(fig)