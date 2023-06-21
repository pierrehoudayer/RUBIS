import matplotlib        as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcl
import numpy             as np
from matplotlib             import rc, ticker
from matplotlib.collections import LineCollection
from pylab                  import cm

from helpers                import find_domains
from legendre               import pl_eval_2D, pl_project_2D

def phi_g_harmonics(zeta, phi_g_l, cmap=cm.viridis, radial=True) : 
    """
    Displays the gravitational potential harmonics and gives an 
    estimate of the error on Poisoon's equation induced by this 
    decomposition.
    
    Parameters
    ----------
    zeta : array_like, shape (N, )
        Variable labelling the isopotentials
    phi_g_l : array_like, shape (N, L)
        Gravitational potential harmonics.
    cmap : ColorMap instance, optional
        Colormap used to display the harmonics (as a function
        of their degrees). Default is cm.viridis.
    radial : boolean, optional
        True if the harmonics come from the radial method routine
        and false otherwise. Default is True.
    """
    z_max = 1.3
    L = phi_g_l.shape[1]
    if radial :
        # External domain
        z_ext = np.linspace(1.0, z_max, 101)
        zeta = np.concatenate((zeta, z_ext))
    
        # Definition of all harmonics
        phi_g_l = np.vstack((
            phi_g_l,
            phi_g_l[-1] * (z_ext[:, None])**-(np.arange(L)+1)
        ))
    else : 
        phi_g_l = phi_g_l[zeta < z_max]
        zeta = zeta[zeta < z_max]
        
    # Error on Poisson's equation
    Poisson_error = np.abs(phi_g_l[:, -1]/phi_g_l[:, 0]).max()
    print(f"Estimated error on Poisson's equation: {round(Poisson_error, 16)}")
    
    # Plot
    ylims = (1e-23, 1e2)
    for l in range(0, L, 2):
        c = cmap(l/L)
        plt.plot(zeta, np.abs(phi_g_l[:, l]), color=c, lw=1.0, alpha=0.3)
    plt.vlines(
        find_domains(zeta).bounds, 
        ymin=ylims[0],  ymax=ylims[1], colors='grey', linestyles='--', linewidth=1.0
    )
    plt.yscale('log')
    plt.ylim(*ylims)
    plt.show()
    
def get_cmap_from_proplot(cmap_name, **kwargs) :
    '''
    Get a colormap defined in the proplot extension. If proplot 
    isn't installed, then return a matplotlib colormap corresponding
    to cmap_name.
    
    Parameters
    ----------
    cmap_name: string
        String corresponding to the colormap name in proplot.
        
    Returns
    -------
    cmap: Colormap instance
        Corresponding colormap.
    '''
    from importlib.util import find_spec
    spec = find_spec('proplot')
    
    if spec is None :  # proplot is not installed
        try : 
            cmap = cm.get_cmap(cmap_name)
        except : 
            stellar_list = ["#fffffe", "#f6cf77", "#bd7a37", "#6a1707", "#1d1d1d"][::-1]
            # fire_list    = ["#fffdfb", "#f7be7a", "#d96644", "#8f3050", "#401631"][::-1]
            cmap = get_continuous_cmap(stellar_list)
        return cmap
    else :             # proplot is installed
        import proplot as pplt
        cmap = pplt.Colormap(cmap_name, **kwargs)
        return cmap
    
def hex_to_rgb(hex_value) :
    '''
    Converts hex to rgb colours
    
    Parameters
    ----------
    hex_value: string
        String of 6 characters representing a hex colour
        
    Returns
    -------
    rgb_values: tuple
        Lenght 3 list of RGB values
    '''
    hex_value = hex_value.strip("#") # removes hash symbol if present
    lv = len(hex_value)
    rgb_values = tuple(
        int(hex_value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)
    )
    return rgb_values


def rgb_to_dec(rgb_values) :
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    
    Parameters
    ----------
    rgb_values: tuple of integers
        Lenght 3 tuple with RGB values
        
    Returns
    -------
    dec_values: tuple of floats
        Lenght 3 tuple with decimal values
    '''
    dec_values = [v/256 for v in rgb_values]
    return dec_values

def get_continuous_cmap(hex_list, float_list=None):
    '''
    Creates and returns a color map that can be used in heat map figures.
    If float_list is not provided, colour map graduates 
    linearly between each color in hex_list. If float_list is provided, 
    each color in hex_list is mapped to the respective location in float_list. 
    
    Parameters
    ----------
    hex_list: list of strings
        List of hex code strings
    float_list: list of floats
        List of floats between 0 and 1, same length as hex_list.
        Must start with 0 and end with 1.
        
    Returns
    -------
    cmap: Colormap instance
        Colormap
    '''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [
            [float_list[i], rgb_list[i][num], rgb_list[i][num]] 
            for i in range(len(float_list))
        ]
        cdict[col] = col_list
    cmap = mcl.LinearSegmentedColormap('my_cmap', segmentdata=cdict, N=256)
    return cmap

def plot_flux_lines(r, t, **kwargs) : 
    """
    Return the fig and axes with the flux lines plotted upon.

    Parameters
    ----------
    r : array_like, shape (N, M)
        Radius values for each line.
    f_l : array_like, shape (N, M)
        Angular (cos theta) values along the radius for each line.
    kwargs : 
        keyboard arguments to be passed to plt.plot()

    Returns
    -------
    None.

    """
    # Define the limit angles
    r0, t0 = r[0] , t[0]
    r1, t1 = r[-1], t[-1]
    s0, s1 = (1-t0**2)**0.5, (1-t1**2)**0.5
    
    # Initialise the figure
    margin, cbar_width = 0.05, 0.1
    x_scale = 2 * margin + np.abs(r1 * s1).max() + 2 * cbar_width
    y_scale = 2 * margin + np.abs(r1 * t1).max()
    factor = min(18/x_scale, 9.5/y_scale)
    fig, ax = plt.subplots(figsize=(x_scale * factor, y_scale * factor), frameon=False)
    
    # Plot the constant flux surface
    x0_long = np.hstack(((r0*s0)[::-1], -r0*s0, (r0*s0)[-1]))
    y0_long = np.hstack(((r0*t0)[::-1],  r0*t0, (r0*t0)[-1]))
    ax.plot(x0_long, y0_long, lw=1.0, **kwargs)
    
    # Plot the characteristics
    for rk, tk in zip(r.T, t.T) :
        tk = np.where(tk > 1.0, 1.0, tk)
        sk = (1-tk**2)**0.5
        for sgn in [1, -1] : 
            ax.plot(sgn*rk*sk, rk*tk, lw=0.5, alpha=0.5, zorder=10, **kwargs)
    return (fig, ax)

def set_axes_equal(ax) :
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Parameters
    ----------
      ax: matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = max([x_range, y_range, z_range]) / 3

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
def plot_3D_surface(surf_l, f_l, show_T_eff, res, cmap) : 
    """
    Create a 3D plot of the star's surface, colored by the values 
    of f.

    Parameters
    ----------
    surf_l : array_like, shape (L, )
        Surface mapping harmonics.
    f_l : array_like, shape (L, )
        Function values harmonics on the surface.
    show_T_eff : boolean, optional
        Whether to map the effective T_eff instead of the radiative flux
        amplitude on the surface.       
    res : tuple of floats (res_t, res_p)
        Gives the resolution of the 3D surface in theta and phi coordinates 
        respectively.
    cmap : ColorMap instance, optional
        Colormap used to display the f values on the surface.

    Returns
    -------
    None.

    """
    # 1D variables
    res_t, res_p = res
    t = np.linspace(-1, 1, res_t)
    r = pl_eval_2D(surf_l, t)
    if show_T_eff : 
        f = np.abs(pl_eval_2D(f_l, t)) ** 0.25
        title = (
              r"$\displaystyle T_\mathrm{eff} \times "
            + r"\left[\frac{L}{4\pi \sigma {R_\mathrm{eq}}^2}\right]^{-1/4}$"
        )
    else : 
        f = pl_eval_2D(f_l, t)
        title = r"$\displaystyle Q \times \left[\frac{L}{4\pi {R_\mathrm{eq}}^2}\right]^{-1}$"
    s, z = r * (1-t**2)**0.5, r * t
    p = np.linspace(0, 2*np.pi, res_p)

    # 2D variables
    X = s[:, None] * np.cos(p) 
    Y = s[:, None] * np.sin(p)
    Z = np.tile(z, (res_p, 1)).T
    F = np.tile(f, (res_p, 1)).T
    
    # Colormap
    stellar_list = ["#fffffe", "#f6cf77", "#bd7a37", "#6a1707", "#1d1d1d"][::-1]
    if cmap is None : cmap = get_continuous_cmap(stellar_list)

    # 3D Plot
    fig = plt.figure(figsize=(10.0, 10.0))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1.0, 1.0, 1.0])
    ax_surf = ax.plot_surface(
        X, Y, Z, facecolors=cmap(F/f.max()), shade=False, 
        rcount=max(res_p, res_t), ccount=max(res_p, res_t)
    )
    ax_surf.set_edgecolor((1.0, 1.0, 1.0, 0.1))
    ax_surf.set_linewidth(0.1)
    ax.set_axis_off()
    set_axes_equal(ax)
    ax.view_init(-150, 0)
    
    # Colorbar
    rc('text', usetex=True)
    cbar_width, size, ticks = 0.1, 20, [0, 10**int(np.log10(f.max()))]
    cbr = fig.colorbar(
        mpl.cm.ScalarMappable(norm=mcl.Normalize(vmax=f.max(), vmin=0.0), cmap=cmap), 
        ticks=ticks, pad=0.0, fraction=cbar_width, shrink=0.6, aspect=25, extend="max"
    )
    cbr.ax.set_title(title, y=1.07, fontsize=size)
    cbr.ax.tick_params(labelsize=size)
    
    # Showing the figure
    fig.tight_layout()
    plt.show()

def plot_f_map(
    map_n, f, phi_eff, max_degree,
    angular_res=501, t_deriv=0, levels=100, cmap=cm.Blues, size=16, label=r"$f$",
    show_surfaces=False, n_lines=30, cmap_lines=cm.BuPu, lw=0.5,
    disc=None, disc_color='white', map_ext=None, n_lines_ext=20,
    add_to_fig=None, background_color='white',
) :
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
        Color used to display the discontinuities. The default is 'white'.
    map_ext : array_like, shape (Ne, M), optional
        Used to show the external mapping, if given.
    n_lines_ext : integer, optional
        Number of level surfaces in the external mapping. The default is 20.
    add_to_fig : fig object, optional
        If given, the figure on which the plot should be added. 
        The default is None.
    background_color : string, optional
        Optional color for the plot background. The default is 'white'.

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
    rc('text', usetex=True)
    rc('xtick', labelsize=size)
    rc('ytick', labelsize=size)
    rc('axes', facecolor=background_color)
    
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
    
    # Right side
    csr = ax.contourf(
        map_res[N-Nf:]*sth_res, map_res[N-Nf:]*cth_res, f2D, 
        cmap=cmap, norm=norm, levels=levels
    )
    for c in csr.collections:
        c.set_edgecolor("face")
    if disc is not None :
        for i in disc :
            plt.plot(map_res[i]*sth_res, map_res[i]*cth_res, color=disc_color, lw=lw)
    plt.plot(map_res[-1]*sth_res, map_res[-1]*cth_res, 'k-', lw=lw)
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
            ls, location='left', pad=cbar_width, fraction=cbar_width, shrink=0.85, aspect=25
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
                plt.plot(-map_res[i]*sth_res, map_res[i]*cth_res, 'w-', lw=lw)
        plt.plot(-map_res[-1]*sth_res, map_res[-1]*cth_res, 'k-', lw=lw)
        
    # External mapping
    if map_ext is not None : 
        Ne, _ = map_ext.shape
        map_ext_l   = pl_project_2D(map_ext, max_degree)
        map_ext_res = pl_eval_2D(map_ext_l, np.linspace(-1, 1, angular_res))
        for ri in map_ext_res[::-Ne//n_lines_ext] : 
            plt.plot( ri*sth_res, ri*cth_res, lw=lw/2, ls='-', color='grey')
            plt.plot(-ri*sth_res, ri*cth_res, lw=lw/2, ls='-', color='grey')
    
    # Show figure
    plt.axis('equal')
    plt.xlabel('$s/R_\mathrm{eq}$', fontsize=size+3)
    plt.ylabel('$z/R_\mathrm{eq}$', fontsize=size+3)
    plt.xlim((-1.0, 1.0))
    fig.tight_layout()
    plt.show()