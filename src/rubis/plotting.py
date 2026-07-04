"""Visualisation utilities for RUBIS models."""

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from .flux import RadiativeFlux
from .legendre import pl_eval_2D, pl_project_2D
from .models import Model2D, VacuumModel2D

__all__ = [
    "STELLAR_CMAP",
    "STELLAR_CMAP_R",
    "plot_gravitational_harmonics",
    "plot_model_field",
    "plot_radiative_flux_lines",
    "plot_radiative_flux_surface",
]

ColormapLike = str | mcolors.Colormap | None

_STELLAR_COLORS = (
    "#1d1d1d",
    "#6a1707",
    "#bd7a37",
    "#f6cf77",
    "#fffffe",
)

STELLAR_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "stellar", _STELLAR_COLORS
)
STELLAR_CMAP_R = STELLAR_CMAP.reversed(name="stellar_r")


def _resolve_colormap(cmap: ColormapLike) -> mcolors.Colormap:
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
        raise ValueError(f"Unknown colormap {cmap!r}.") from exc


def _resolve_max_degree(
    model: Model2D,
    max_degree: int | None,
) -> int:
    """Return a valid angular truncation degree."""
    L = model.angular_resolution if max_degree is None else max_degree
    if not 1 <= L <= model.angular_resolution:
        raise ValueError(
            "max_degree must lie between 1 and the model "
            "angular resolution."
        )
    return L


def _evaluate_field(
    model: Model2D,
    field: ArrayLike,
    t: np.ndarray,
    max_degree: int,
    angular_derivative: int,
) -> np.ndarray:
    """Evaluate a material field on a regular angular grid."""
    field = np.asarray(field)

    if field.shape == (model.n_points,):
        if angular_derivative != 0:
            raise ValueError(
                "A one-dimensional field has no angular derivative."
            )
        return np.repeat(field[:, None], t.size, axis=1)

    if field.shape != model.r2d.shape:
        raise ValueError(
            "field must have shape (model.n_points,) "
            "or model.r2d.shape."
        )
    if angular_derivative not in (0, 1, 2):
        raise ValueError("angular_derivative must be 0, 1, or 2.")

    field_l = pl_project_2D(field, max_degree, even=False)
    evaluated = pl_eval_2D(field_l, t, der=angular_derivative)
    return (
        evaluated if angular_derivative == 0
        else evaluated[angular_derivative]
    )


def _sample_indices(size: int, count: int) -> np.ndarray:
    """Return approximately uniform indices from the outer to inner boundary."""
    if count < 1:
        raise ValueError(
            "The number of plotted surfaces must be positive."
        )
    return np.unique(
        np.linspace(0, size - 1, min(size, count), dtype=int)
    )


def _set_axes_equal(ax: Axes) -> None:
    """Give all three axes the same displayed extent."""
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    centers = limits.mean(axis=1)
    radius = np.ptp(limits, axis=1).max() / 2.0

    ax.set_xlim3d(centers[0] - radius, centers[0] + radius)
    ax.set_ylim3d(centers[1] - radius, centers[1] + radius)
    ax.set_zlim3d(centers[2] - radius, centers[2] + radius)


def plot_gravitational_harmonics(
    model: Model2D,
    *,
    vacuum: VacuumModel2D | None = None,
    max_degree: int | None = None,
    cmap: ColormapLike = "viridis",
    y_limits=(1.0e-22, 1.0e2),
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot the Legendre harmonics of the gravitational potential."""
    L = _resolve_max_degree(model, max_degree)
    zeta = model.zeta
    phi_g_l = pl_project_2D(model.phi_g, L)

    interfaces = list(model.domains.interface_values)
    if vacuum is not None:
        if vacuum.angular_resolution < L:
            raise ValueError(
                "vacuum angular resolution is smaller than max_degree."
            )
        zeta = np.hstack((zeta, vacuum.zeta[1:]))
        phi_g_l = np.vstack((
            phi_g_l,
            pl_project_2D(vacuum.phi_g, L)[1:],
        ))
        interfaces.append(model.zeta[-1])

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    cmap = _resolve_colormap(cmap)
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
        interfaces,
        ymin=y_limits[0],
        ymax=y_limits[1],
        colors="grey",
        linestyles="--",
        linewidth=1.0,
    )
    ax.set_yscale("log")
    ax.set_ylim(*y_limits)
    ax.set_xlabel(r"$\zeta$")
    ax.set_ylabel(r"$|\phi_{g,\ell}|$")
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
    r = np.asarray(flux.line_r)
    t = np.clip(np.asarray(flux.line_t), -1.0, 1.0)

    r0, r1 = r[0], r[-1]
    t0, t1 = t[0], t[-1]
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
        y_scale = 2.0 * margin + np.abs(r1 * t1).max()
        factor = min(18.0 / x_scale, 9.5 / y_scale)
        fig, ax = plt.subplots(
            figsize=(x_scale * factor, y_scale * factor),
            frameon=False,
        )
    else:
        fig = ax.figure

    x0 = np.hstack((
        (r0 * s0)[::-1],
        -r0 * s0,
        (r0 * s0)[-1],
    ))
    y0 = np.hstack((
        (r0 * t0)[::-1],
        +r0 * t0,
        (r0 * t0)[-1],
    ))
    ax.plot(x0, y0, linewidth=1.0, **line_options)

    for r_line, t_line in zip(r.T, t.T):
        s_line = np.sqrt(1.0 - t_line**2)
        for sign in (-1.0, 1.0):
            ax.plot(
                sign * r_line * s_line,
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


def _plot_axisymmetric_surface(
    surface_l,
    field_l,
    *,
    effective_temperature,
    resolution,
    cmap: ColormapLike,
    ax: Axes | None,
    font_size,
) -> tuple[Figure, Axes]:
    """Plot an axisymmetric surface coloured by a scalar field."""
    res_t, res_p = resolution
    t = np.linspace(-1.0, 1.0, res_t)
    r = pl_eval_2D(surface_l, t)

    if effective_temperature:
        field = np.abs(pl_eval_2D(field_l, t)) ** 0.25
        title = (
            r"$T_{\rm eff}\,"
            r"\left[\frac{L}{4\pi\sigma R_{\rm eq}^2}"
            r"\right]^{-1/4}$"
        )
    else:
        field = pl_eval_2D(field_l, t)
        title = (
            r"$Q\,\left[\frac{L}{4\pi R_{\rm eq}^2}"
            r"\right]^{-1}$"
        )

    s = r * np.sqrt(1.0 - t**2)
    z = r * t
    p = np.linspace(0.0, 2.0 * np.pi, res_p)

    x = s[:, None] * np.cos(p)
    y = s[:, None] * np.sin(p)
    z2d = np.broadcast_to(z[:, None], x.shape)
    field2d = np.broadcast_to(field[:, None], x.shape)

    if ax is None:
        fig = plt.figure(figsize=(10.0, 10.0))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    cmap = _resolve_colormap(cmap)
    vmax = float(np.nanmax(field))
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
        facecolors=cmap(norm(field2d)),
        shade=False,
        rcount=max(res_p, res_t),
        ccount=max(res_p, res_t),
    )
    surface.set_edgecolor((1.0, 1.0, 1.0, 0.1))
    surface.set_linewidth(0.1)

    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_axis_off()
    _set_axes_equal(ax)
    ax.view_init(elev=-150, azim=0)

    scalar_map = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_map.set_array([])
    colorbar = fig.colorbar(
        scalar_map,
        ax=ax,
        pad=0.0,
        fraction=0.1,
        shrink=0.6,
        aspect=25,
    )
    colorbar.ax.set_title(title, y=1.07, fontsize=font_size)
    colorbar.ax.tick_params(labelsize=font_size)
    fig.tight_layout()

    return fig, ax


def plot_radiative_flux_surface(
    model: Model2D,
    flux: RadiativeFlux,
    *,
    show_effective_temperature=True,
    resolution=(200, 100),
    cmap: ColormapLike = "magma_r",
    ax: Axes | None = None,
    font_size=16,
) -> tuple[Figure, Axes]:
    """Plot the radiative-flux distribution on the stellar surface."""
    _resolve_max_degree(model, flux.max_degree)
    surface_l = pl_project_2D(model.r2d[-1], flux.max_degree)
    return _plot_axisymmetric_surface(
        surface_l,
        flux.surface_flux_l,
        effective_temperature=show_effective_temperature,
        resolution=resolution,
        cmap=cmap,
        ax=ax,
        font_size=font_size,
    )


def plot_model_field(
    model: Model2D,
    field: ArrayLike,
    *,
    vacuum: VacuumModel2D | None = None,
    max_degree: int | None = None,
    angular_resolution=501,
    angular_derivative=0,
    levels=100,
    cmap: ColormapLike = "Blues",
    norm: mcolors.Normalize | None = None,
    label=r"$f$",
    show_surfaces=False,
    surface_count=30,
    surface_cmap: ColormapLike = "BuPu",
    line_width=0.5,
    interface_color="white",
    vacuum_surface_count=20,
    ax: Axes | None = None,
    background_color="white",
    font_size=16,
) -> tuple[Figure, Axes]:
    """Plot a scalar field on a converged material model."""
    if angular_resolution < 2:
        raise ValueError("angular_resolution must be at least 2.")

    L = _resolve_max_degree(model, max_degree)
    t = np.linspace(-1.0, 1.0, angular_resolution)
    s = np.sqrt(1.0 - t**2)

    r_l = pl_project_2D(model.r2d, L)
    r = pl_eval_2D(r_l, t)
    field2d = _evaluate_field(
        model, field, t, L, angular_derivative
    )

    cmap = _resolve_colormap(cmap)
    surface_cmap = _resolve_colormap(surface_cmap)

    if ax is None:
        margin = 0.05
        colorbar_width = 0.1
        x_scale = (
              2.0 * margin
            + np.abs(r[-1] * s).max()
            + 4.0 * colorbar_width
        )
        y_scale = 2.0 * margin + np.abs(r[-1] * t).max()
        factor = min(18.0 / x_scale, 9.5 / y_scale)
        fig, ax = plt.subplots(
            figsize=(x_scale * factor, y_scale * factor),
            frameon=False,
        )
    else:
        fig = ax.figure

    ax.set_facecolor(background_color)
    ax.tick_params(labelsize=font_size)

    right = ax.contourf(
        r * s,
        r * t,
        field2d,
        cmap=cmap,
        norm=norm,
        levels=levels,
        antialiased=False,
    )

    for i in model.domains.interface_end_indices:
        ax.plot(
            r[i] * s,
            r[i] * t,
            color=interface_color,
            linewidth=line_width,
        )
    ax.plot(r[-1] * s, r[-1] * t, color="black", linewidth=line_width)

    colorbar_width = 0.1
    tick_locator = ticker.MaxNLocator(nbins=5)
    colorbar = fig.colorbar(
        right,
        ax=ax,
        pad=0.7 * colorbar_width,
        fraction=colorbar_width,
        shrink=0.85,
        aspect=25,
    )
    colorbar.locator = tick_locator
    colorbar.update_ticks()
    colorbar.ax.set_title(label, y=1.03, fontsize=font_size + 3)

    if show_surfaces:
        indices = _sample_indices(model.n_points, surface_count)
        segments = [
            np.column_stack((-r[i] * s, r[i] * t))
            for i in indices
        ]
        surfaces = LineCollection(
            segments,
            cmap=surface_cmap,
            linewidths=line_width,
        )
        surfaces.set_array(model.phi_eff[indices])
        ax.add_collection(surfaces)

        surface_bar = fig.colorbar(
            surfaces,
            ax=ax,
            location="left",
            pad=colorbar_width,
            fraction=colorbar_width,
            shrink=0.85,
            aspect=25,
        )
        surface_bar.locator = tick_locator
        surface_bar.update_ticks()
        surface_bar.ax.set_title(
            r"$\phi_{\rm eff}\,(GM/R_{\rm eq})^{-1}$",
            y=1.03,
            fontsize=font_size + 3,
        )
    else:
        ax.contourf(
            -r * s,
            r * t,
            field2d,
            cmap=cmap,
            norm=norm,
            levels=levels,
            antialiased=False,
        )
        for i in model.domains.interface_end_indices:
            ax.plot(
                -r[i] * s,
                r[i] * t,
                color=interface_color,
                linewidth=line_width,
            )
        ax.plot(
            -r[-1] * s,
            r[-1] * t,
            color="black",
            linewidth=line_width,
        )

    if vacuum is not None:
        if vacuum.angular_resolution < L:
            raise ValueError(
                "vacuum angular resolution is smaller than max_degree."
            )

        r_vac = pl_eval_2D(
            pl_project_2D(vacuum.r2d, L), t
        )[1:]
        r_vac = r_vac[np.all(np.isfinite(r_vac), axis=1)]
        if r_vac.size:
            for i in _sample_indices(
                r_vac.shape[0], vacuum_surface_count
            ):
                ax.plot(
                    r_vac[i] * s,
                    r_vac[i] * t,
                    color="grey",
                    linewidth=line_width / 2.0,
                )
                ax.plot(
                    -r_vac[i] * s,
                    r_vac[i] * t,
                    color="grey",
                    linewidth=line_width / 2.0,
                )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$s/R_{\rm eq}$", fontsize=font_size + 3)
    ax.set_ylabel(r"$z/R_{\rm eq}$", fontsize=font_size + 3)
    ax.set_xlim(-1.0, 1.0)
    fig.tight_layout()

    return fig, ax