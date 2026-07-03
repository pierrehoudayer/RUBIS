"""Visualisation utilities for RUBIS models."""

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from .domains import find_domains
from .flux import RadiativeFlux
from .legendre import (
    pl_eval_2D,
    pl_project_2D,
)
from .models import Model2D


ColormapLike = str | mcolors.Colormap | None


_STELLAR_COLORS = (
    "#1d1d1d",
    "#6a1707",
    "#bd7a37",
    "#f6cf77",
    "#fffffe",
)

STELLAR_CMAP = (
    mcolors.LinearSegmentedColormap.from_list(
        "stellar",
        _STELLAR_COLORS,
    )
)

STELLAR_CMAP_R = STELLAR_CMAP.reversed(
    name="stellar_r"
)


def _resolve_colormap(
    cmap: ColormapLike,
) -> mcolors.Colormap:
    """Resolve a Matplotlib or RUBIS colormap."""
    if cmap is None:
        return STELLAR_CMAP

    if isinstance(cmap, mcolors.Colormap):
        return cmap

    if cmap == "stellar":
        return STELLAR_CMAP

    if cmap == "stellar_r":
        return STELLAR_CMAP_R

    try:
        return mpl.colormaps[cmap]
    except KeyError as exc:
        raise ValueError(
            f"Unknown colormap {cmap!r}."
        ) from exc


def phi_g_harmonics(
    zeta,
    phi_g_l,
    cmap: ColormapLike = "viridis",
    radial=True,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot gravitational-potential harmonics.

    This legacy interface is retained temporarily. The harmonics will
    later be reconstructed directly from Model2D and VacuumModel2D.
    """
    z_max = 1.3
    L = phi_g_l.shape[1]

    if radial:
        # External domain
        z_ext = np.linspace(1.0, z_max, 101)
        zeta = np.concatenate((zeta, z_ext))

        # Analytic vacuum continuation
        phi_g_l = np.vstack((
            phi_g_l,
            phi_g_l[-1] * z_ext[:, None] ** -(np.arange(L) + 1)
        ))

    else:
        inside = zeta < z_max
        phi_g_l = phi_g_l[inside]
        zeta = zeta[inside]

    poisson_error = np.max(
        np.abs(phi_g_l[:, -1] / phi_g_l[:, 0])
    )

    print(f"Estimated error on Poisson's equation: {poisson_error:.3e}")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    cmap = _resolve_colormap(cmap)
    y_limits = (1.0e-22, 1.0e2)

    denominator = max(L - 1, 1)

    for l in range(0, L, 2):
        ax.plot(
            zeta,
            np.abs(phi_g_l[:, l]),
            color=cmap(l / denominator),
            linewidth=1.0,
            alpha=0.3,
        )

    ax.vlines(
        find_domains(zeta).interface_values,
        ymin=y_limits[0],
        ymax=y_limits[1],
        colors="grey",
        linestyles="--",
        linewidth=1.0,
    )

    ax.set_yscale("log")
    ax.set_ylim(*y_limits)
    ax.set_yticks([
        1.0e-20,
        1.0e-15,
        1.0e-10,
        1.0e-5,
        1.0,
    ])

    fig.tight_layout()

    return fig, ax


def plot_flux_lines(
    r,
    t,
    *,
    ax: Axes | None = None,
    font_size=16,
    **line_options,
) -> tuple[Figure, Axes]:
    """Plot radiative-flux characteristics."""
    r = np.asarray(r)
    t = np.clip(np.asarray(t), -1.0, 1.0)

    # Define the limiting surfaces
    r0 = r[0]
    t0 = t[0]
    r1 = r[-1]
    t1 = t[-1]

    s0 = np.sqrt(1.0 - t0**2)
    s1 = np.sqrt(1.0 - t1**2)

    if ax is None:
        margin = 0.05
        colorbar_width = 0.1

        x_scale = (
              2.0 * margin 
            + 2.0 * colorbar_width
            + np.abs(r1 * s1).max()
        )
        y_scale = (
            2.0 * margin
            + np.abs(r1 * t1).max()
        )

        factor = min(18.0 / x_scale, 9.5 / y_scale)

        fig, ax = plt.subplots(
            figsize=(
                x_scale * factor,
                y_scale * factor,
            ),
            frameon=False,
        )
    else:
        fig = ax.figure

    # Constant-flux inner surface
    x0 = np.hstack((
        (r0 * s0)[::-1],
        -r0 * s0,
        (r0 * s0)[-1],
    ))
    y0 = np.hstack((
        (r0 * t0)[::-1],
        r0 * t0,
        (r0 * t0)[-1],
    ))

    ax.plot(
        x0,
        y0,
        linewidth=1.0,
        **line_options,
    )

    # Characteristics
    for r_line, t_line in zip(r.T, t.T):
        s_line = np.sqrt(1.0 - t_line**2)

        for sign in (-1.0, 1.0):
            ax.plot(
                sign
                * r_line
                * s_line,
                r_line * t_line,
                linewidth=0.5,
                alpha=0.5,
                zorder=10,
                **line_options,
            )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$s/R_{\rm eq}$", fontsize=font_size)
    ax.set_ylabel(r"$z/R_{\rm eq}$", fontsize=font_size)
    ax.tick_params(labelsize=font_size)

    fig.tight_layout()

    return fig, ax


def plot_radiative_flux_lines(
    flux: RadiativeFlux,
    *,
    ax: Axes | None = None,
    font_size=16,
    **line_options,
) -> tuple[Figure, Axes]:
    """Plot the characteristics of a radiative-flux solution."""
    return plot_flux_lines(
        flux.line_r,
        flux.line_t,
        ax=ax,
        font_size=font_size,
        **line_options,
    )
    

def set_axes_equal(ax: Axes) -> None:
    """Give all three axes the same displayed extent."""
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    centers = limits.mean(axis=1)
    radius = np.ptp(limits, axis=1).max() / 2.0

    ax.set_xlim3d(
        centers[0] - radius,
        centers[0] + radius,
    )
    ax.set_ylim3d(
        centers[1] - radius,
        centers[1] + radius,
    )
    ax.set_zlim3d(
        centers[2] - radius,
        centers[2] + radius,
    )
    
    
def plot_3D_surface(
    surf_l,
    f_l,
    show_T_eff,
    res,
    cmap: ColormapLike,
    *,
    ax: Axes | None = None,
    font_size=16,
) -> tuple[Figure, Axes]:
    """Plot an axisymmetric surface coloured by a scalar field."""
    res_t, res_p = res

    t = np.linspace(-1.0, 1.0, res_t)
    r = pl_eval_2D(surf_l, t)

    if show_T_eff:
        f = np.abs(pl_eval_2D(f_l, t)) ** 0.25

        title = (
            r"$T_{\rm eff}\,"
            r"\left["
            r"\frac{L}"
            r"{4\pi\sigma R_{\rm eq}^2}"
            r"\right]^{-1/4}$"
        )

    else:
        f = pl_eval_2D(f_l, t)

        title = (
            r"$Q\,"
            r"\left["
            r"\frac{L}"
            r"{4\pi R_{\rm eq}^2}"
            r"\right]^{-1}$"
        )

    s = r * np.sqrt(1.0 - t**2)
    z = r * t

    p = np.linspace(0.0, 2.0 * np.pi, res_p)

    x = s[:, None] * np.cos(p)
    y = s[:, None] * np.sin(p)
    z2d = np.broadcast_to(z[:, None], x.shape)
    f2d = np.broadcast_to(f[:, None], x.shape)

    if ax is None:
        fig = plt.figure(
            figsize=(10.0, 10.0)
        )
        ax = fig.add_subplot(
            111,
            projection="3d",
        )
    else:
        fig = ax.figure

    cmap = _resolve_colormap(cmap)
    vmax = float(np.nanmax(f))

    if not np.isfinite(vmax):
        raise ValueError(
            "The surface field contains no finite maximum."
        )

    if vmax <= 0.0:
        raise ValueError(
            "The surface field must contain positive values."
        )

    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)

    surface = ax.plot_surface(
        x,
        y,
        z2d,
        facecolors=cmap(norm(f2d)),
        shade=False,
        rcount=max(res_p, res_t),
        ccount=max(res_p, res_t),
    )
    surface.set_edgecolor((1.0, 1.0, 1.0, 0.1))
    surface.set_linewidth(0.1)

    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_axis_off()
    set_axes_equal(ax)
    ax.view_init(elev=-150, azim=0)

    scalar_map = mpl.cm.ScalarMappable(
        norm=norm,
        cmap=cmap,
    )
    scalar_map.set_array([])

    colorbar = fig.colorbar(
        scalar_map,
        ax=ax,
        pad=0.0,
        fraction=0.1,
        shrink=0.6,
        aspect=25,
    )

    colorbar.ax.set_title(
        title,
        y=1.07,
        fontsize=font_size,
    )
    colorbar.ax.tick_params(labelsize=font_size)

    fig.tight_layout()

    return fig, ax
    
    
def plot_radiative_flux_surface(
    model: Model2D,
    flux: RadiativeFlux,
    *,
    show_effective_temperature=True,
    resolution=(
        200,
        100,
    ),
    cmap: ColormapLike = "magma_r",
    ax: Axes | None = None,
    font_size=16,
) -> tuple[Figure, Axes]:
    """Plot the radiative-flux distribution on the stellar surface."""
    surface_l = pl_project_2D(model.r2d[-1], flux.max_degree)

    return plot_3D_surface(
        surface_l,
        flux.surface_flux_l,
        show_T_eff=show_effective_temperature,
        res=resolution,
        cmap=cmap,
        ax=ax,
        font_size=font_size,
    )
    

def plot_f_map(
    map_n,
    f,
    phi_eff,
    max_degree,
    angular_res=501,
    t_deriv=0,
    levels=100,
    cmap: ColormapLike = "Blues",
    size=16,
    label=r"$f$",
    show_surfaces=False,
    n_lines=30,
    cmap_lines: ColormapLike = "BuPu",
    lw=0.5,
    disc=None,
    disc_color="white",
    map_ext=None,
    n_lines_ext=20,
    add_to_fig=None,
    background_color="white",
):
    """
    Shows the value of f in the 2D model.

    Parameters
    ----------
    map_n : array_like, shape (N, M)
        2D Mapping.
    f : array_like, shape (N, ) or (N, M)
        Function value on the surface levels or at each point on the mapping.
    phi_eff : array_like, shape (N, )
        Value of the effective potential on each isopotential.
        Serves the colormapping if show_surfaces=True.
    max_degree : integer
        number of harmonics to use for interpolating the mapping.
    angular_res : integer, optional
        angular resolution used to plot the mapping. The default is 501.
    t_deriv : integer, optional
        derivative (with respect to t = cos(theta)) order to plot. Only used
        is len(f.shape) == 2. The default is 0.
    levels : integer, optional
        Number of color levels on the plot. The default is 100.
    cmap : cm.cmap instance, optional
        Colormap for the plot. The default is cm.Blues.
    size : integer, optional
        Fontsize. The default is 16.
    label : string, optional
        Name of the f variable. The default is r"$f$"
    show_surfaces : boolean, optional
        Show the isopotentials on the left side if set to True.
        The default is False.
    n_lines : integer, optional
        Number of equipotentials on the plot. The default is 50.
    cmap_lines : cm.cmap instance, optional
        Colormap used for the isopotential plot. 
        The default is cm.BuPu.
    disc : array_like, shape (Nd, ), optional
        Indices of discontinuities to plot. The default is None.
    disc_color : string, optional
        Color used to display the discontinuities. The default is "white".
    map_ext : array_like, shape (Ne, M), optional
        Used to show the external mapping, if given.
    n_lines_ext : integer, optional
        Number of level surfaces in the external mapping. The default is 20.
    add_to_fig : fig object, optional
        If given, the figure on which the plot should be added. 
        The default is None.
    background_color : string, optional
        Optional color for the plot background. The default is "white".

    Returns
    -------
    None.

    """
    
    # Angular interpolation
    N, M = map_n.shape
    cth_res = np.linspace(-1, 1, angular_res)
    sth_res = np.sqrt(1-cth_res**2)
    map_l   = pl_project_2D(map_n, max_degree)
    map_res = pl_eval_2D(map_l, cth_res)
    
    # 2D density
    if len(f.shape) == 1 :
        f2D = np.tile(f, angular_res).reshape((angular_res, -1)).T
    else : 
        f_l = pl_project_2D(f, max_degree, even=False)
        f2D = np.atleast_3d(np.array(pl_eval_2D(f_l, cth_res, der=t_deriv)).T).T[-1]
    Nf = f2D.shape[0]
        
    # Text formating 
    cmap = _resolve_colormap(cmap)
    cmap_lines = _resolve_colormap(cmap_lines)
    norm = None
    midpoint = np.asarray(cmap(0.5)[:3])
    if np.sum((1.0 - midpoint) ** 2) < 1.0e-2:
        norm = mcolors.CenteredNorm()
    
    # Init figure
    norm = None
    if sum((1.0-np.array(cm.get_cmap(cmap)(0.5)[:3]))**2) < 1e-2 : # ~ Test if the cmap is divergent
        norm = mcl.CenteredNorm()
    cbar_width = 0.1
    if add_to_fig is None : 
        margin = 0.05
        x_scale = 2 * margin + (map_res[-1]*sth_res).max() + 4 * cbar_width
        y_scale = 2 * margin + (map_res[-1]*cth_res).max()
        factor = min(18/x_scale, 9.5/y_scale)
        fig, ax = plt.subplots(figsize=(x_scale * factor, y_scale * factor), frameon=False)
    else : 
        fig, ax = add_to_fig
    ax.set_facecolor(background_color)
    ax.tick_params(labelsize=size)
    
    # Right side
    csr = ax.contourf(
        map_res[N-Nf:]*sth_res, map_res[N-Nf:]*cth_res, f2D, 
        cmap=cmap, norm=norm, levels=levels
    )
    for c in csr.collections:
        c.set_edgecolor("face")
    if disc is not None :
        for i in disc :
            ax.plot(map_res[i]*sth_res, map_res[i]*cth_res, color=disc_color, lw=lw)
    ax.plot(map_res[-1]*sth_res, map_res[-1]*cth_res, "k-", lw=lw)
    cbr = fig.colorbar(csr, pad=0.7*cbar_width, fraction=cbar_width, shrink=0.85, aspect=25)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cbr.locator = tick_locator
    cbr.update_ticks()
    cbr.ax.set_title(label, y=1.03, fontsize=size+3)
    
    # Left side
    if show_surfaces :
        ls = LineCollection(
            [np.column_stack([x, y]) for x, y in zip(
                -map_res[::-N//n_lines]*sth_res, 
                 map_res[::-N//n_lines]*cth_res
            )], 
            cmap=cmap_lines, 
            linewidths=lw
        )
        ls.set_array(phi_eff[::-N//n_lines])
        ax.add_collection(ls)
        cbl = fig.colorbar(
            ls, location="left", pad=cbar_width, fraction=cbar_width, shrink=0.85, aspect=25
        )
        cbl.locator = tick_locator
        cbl.update_ticks()
        cbl.ax.set_title(
            r"$\phi_\mathrm{eff} \times \left(GM/R_\mathrm{eq}\right)^{-1}$", 
            y=1.03, fontsize=size+3
        )
    else : 
        csl = ax.contourf(
            -map_res[N-Nf:]*sth_res, map_res[N-Nf:]*cth_res, f2D, 
            cmap=cmap, norm=norm, levels=levels
        )
        for c in csl.collections:
            c.set_edgecolor("face")
        if disc is not None :
            for i in disc :
                ax.plot(-map_res[i]*sth_res, map_res[i]*cth_res, "w-", lw=lw)
        ax.plot(-map_res[-1]*sth_res, map_res[-1]*cth_res, "k-", lw=lw)
        
    # External mapping
    if map_ext is not None : 
        Ne, _ = map_ext.shape
        map_ext_l   = pl_project_2D(map_ext, max_degree)
        map_ext_res = pl_eval_2D(map_ext_l, np.linspace(-1, 1, angular_res))
        for ri in map_ext_res[::-Ne//n_lines_ext] : 
            ax.plot( ri*sth_res, ri*cth_res, lw=lw/2, ls="-", color="grey")
            ax.plot(-ri*sth_res, ri*cth_res, lw=lw/2, ls="-", color="grey")
    
    # Adjust figure
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$s/R_{\rm eq}$", fontsize=size + 3)
    ax.set_ylabel(r"$z/R_{\rm eq}$", fontsize=size + 3)
    ax.set_xlim(-1.0, 1.0)
    fig.tight_layout()

    return fig, ax