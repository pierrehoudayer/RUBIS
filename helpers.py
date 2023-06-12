import matplotlib.pyplot as plt
import matplotlib.colors as mcl
import numpy             as np
from matplotlib             import rc, ticker
from matplotlib.collections import LineCollection
from pylab                  import cm
from scipy.special          import roots_legendre

from legendre               import pl_eval_2D, pl_project_2D

class DotDict(dict):  
    """
    Class that defines dictionaries with dot attributes.
    """   
    def __getattr__(*args):        
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val     
    __setattr__ = dict.__setitem__     
    __delattr__ = dict.__delitem__ 
    
def assign_method(method_choice, model_choice, radial_method, spheroidal_method) : 
    """
    Function assigning the method function to call to a given
    method_choice.

    Parameters
    ----------
    method_choice : string in {'auto', 'radial', 'spheroidal'}
        Method choice (cf. RUBIS.py)
    model_choice : string or DotDict instance
        Model choice (cf. RUBIS.py)
    radial_method : func 
        Function to call if model_choice is set to 'radial'
    spheroidal_method : func 
        Function to call if model_choice is set to 'spheroidal'

    Returns
    -------
    method_func : func in {radial_method, spheroidal_method}
        method function to call for the model deformation.
    """
    
    # Dealing with method_choice = 'auto'
    assert method_choice in {'auto', 'radial', 'spheroidal'}
    if method_choice == 'auto' :
        if isinstance(model_choice, DotDict) :       
            # Checking the number of domains in the composite polytrope 
            if len(np.atleast_1d(model_choice.indices)) > 1 : 
                method_choice = 'spheroidal'
            else : 
                method_choice = 'radial'
        else : 
            # Reading the file 
            radial_coordinate, *_ = np.genfromtxt(
                './Models/'+model_choice, skip_header=2, unpack=True
            )
            if find_domains(radial_coordinate).Nd > 1 :            
                method_choice = 'spheroidal'
            else : 
                method_choice = 'radial'
                
    # Assigning the adaquate method to method_choice
    if method_choice == 'radial' : 
        method_func = radial_method
    else : 
        method_func = spheroidal_method
    return method_func

def give_me_a_name(model_choice, rotation_target) : 
    """
    Constructs a name for the save file using the model name
    and the rotation target.

    Parameters
    ----------
    model_choice : string or Dotdict instance.
        File name or composite polytrope caracteristics.
    rotation_target : float
        Final rotation rate on the equator.

    Returns
    -------
    save_name : string
        Output file name.

    """
    radical = (
        'poly_|' + ''.join(
            str(np.round(index, 1))+"|" for index in np.atleast_1d(model_choice.indices)
        )
        if isinstance(model_choice, DotDict) 
        else model_choice.split('.txt')[0]
    )
    save_name = radical + '_deform_' + str(rotation_target) + '.txt'
    return save_name

def init_2D(r, M) :
    """
    Init function for the angular domain.

    Parameters
    ----------
    r : array_like, shape (N, ) 
        Radial coordinate from the 1D model.
    M : integer
        Angular resolution.

    Returns
    -------
    cth : array_like, shape (M, )
        Angular coordinate (equivalent to cos(theta)).
    map_n : array_like, shape (N, M)
        Isopotential mapping 
        (given by r(phi_eff, theta) = r for now).
    """
    map_n = np.tile(r, (M, 1)).T
    cth, _ = roots_legendre(M)
    return map_n, cth

def init_phi_c(rotation_profile, central_diff_rate, rotation_scale) : 
    """
    Defines the functions used to compute the centrifugal potential
    and the rotation profile with the adequate arguments.
    
    Parameters
    ----------
    rotation_profile : function(r, cth, omega, *args)
        Function used to compute the centrifugal potential, given 
        adequate additional arguments.
    central_diff_rate : float
        Parameter that may be used to compute the centrifugal potential
    rotation_scale : float
        Parameter that may be used to compute the centrifugal potential

    Returns
    -------
    phi_c : function(r, cth, omega)
        Centrifugal potential
    w : function(r, cth, omega)
        Rotation profile

    """
    nb_args = (
          rotation_profile.__code__.co_argcount 
        - len(rotation_profile.__defaults__ or '')
    )
    mask = np.array([0, 1]) < nb_args - 3
    
    # Creation of the centrifugal potential function
    args_phi = np.array([central_diff_rate, rotation_scale])[mask]
    phi_c = lambda r, cth, omega : rotation_profile(r, cth, omega, *args_phi)
    
    # Creation of the rotation profile function
    args_w = np.hstack((np.atleast_1d(args_phi), (True,)))
    w = lambda r, cth, omega : rotation_profile(r, cth, omega, *args_w)
    return phi_c, w

def find_domains(var) :
    """
    Defines many tools to help the domain manipulation and navigation.
    
    Parameters
    ----------
    var : array_like, shape (Nvar, )
        Variable used to define the domains

    Returns
    -------
    dom : DotDict instance.
        Domains informations : {
            Nd : integer
                Number of domains.
            bounds : array_like, shape (Nd-1, )
                Zeta values at boundaries.
            interfaces : list of tuple
                Successives indices of domain interfaces
            beg, end : array_like, shape (Nd-1, ) of integer
                First (resp. last) domain indices.
            edges : array_like, shape (Nd+1, ) of integer
                All edge indices (corresponds to beg + origin + last).
            ranges : list of range()
                All domain index ranges.
            sizes : list of integers
                All domain sizes
            id : array_like, shape (Nvar, ) of integer
                Domain identification number. 
                /!\ if var is zeta, the Nvar = N+Ne!
            id_val : array_like, shape (Nd, ) of integer
                The id values.
            int, ext : array_like, shape (Nvar, ) of boolean
                Interior (resp. exterior, i.e. if rho = 0) domain.
            unq : array_like, shape (Nvar-(Nd-1), ) of integer
                Unique indices through the domains.
            }

    """
    dom = DotDict()
    Nvar = len(var)
    disc = True
    
    # Domain physical boundaries
    unq, unq_idx, unq_inv, unq_cnt = np.unique(
        np.round(var, 15), return_index=True, return_inverse=True, return_counts=True
    )
    cnt_mask = unq_cnt > 1
    dom.bounds = unq[cnt_mask]
    if len(dom.bounds) == 0 : disc = False
    
    # Domain interface indices
    cnt_idx, = np.nonzero(cnt_mask)
    idx_mask = np.in1d(unq_inv, cnt_idx)
    idx_idx, = np.nonzero(idx_mask)
    srt_idx  = np.argsort(unq_inv[idx_mask])
    dom.interfaces = np.split(
        idx_idx[srt_idx], np.cumsum(unq_cnt[cnt_mask])[:-1]
    )
    if disc : dom.end, dom.beg = np.array(dom.interfaces).T
    
    # Domain ranges and sizes
    dom.unq    = unq_idx
    dom.Nd     = len(dom.bounds) + 1
    if disc :
        dom.edges  = np.array((0, ) + tuple(dom.beg) + (Nvar, ))
    else :
        dom.edges  = np.array((0, Nvar, ))
    dom.ranges = list(map(range, dom.edges[:-1], dom.edges[1:]))
    dom.sizes  = list(map(len, dom.ranges))

    # Domain indentification
    dom.id      = np.hstack([d*np.ones(S) for d, S in enumerate(dom.sizes)])
    dom.id_val  = np.unique(dom.id)
    dom.ext     = dom.id == dom.Nd - 1
    dom.int     = np.invert(dom.ext)
    dom.unq_int = np.unique(var[dom.int], return_index=True)[1]
    
    return dom

def valid_reciprocal_domain(x, df, safety=1e-4) :
    """
    Find the valid f domain for a reciprocal function interpolation (i.e. 
    of the function x(f)) knowing df/dx. The function f is allowed to have
    another variable y, in which case the valid domain have the same shape
    as f and is estimated for each value of y.
    
    Parameters
    ----------
    x : array_like, shape (N, )
        Variable along which the f-derivative is taken
    df : array_like, shape (N, ) or shape (N, M)
        Derivative of f with respect to f (the partial derivative w.r.t x should
        correspond to the first axis).

    Returns
    -------
    valid : array_like of boolean, shape (N, ) or shape (N, M)
        Valid domain for the reciprocal function interpolation.
    """
    df = np.atleast_2d(df.T).T
    valid = np.ones_like(df, dtype='bool')
    idx = np.arange(len(x))
    for k, dpk in enumerate(df.T) :
        idx_max = len(idx)
        condition = (dpk < safety) & (x > safety)
        if np.any(condition) : idx_max = np.min(np.argwhere(condition))
        valid[:, k] = (idx < idx_max) & (x > safety)
    valid = np.squeeze(valid)
    return valid

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

def plot_f_map(
    map_n, f, phi_eff, max_degree,
    angular_res=501, t_deriv=0, levels=100, cmap=cm.Blues, size=16, label=r"$f$",
    show_surfaces=False, n_lines=50, cmap_lines=cm.BuPu, lw=0.5,
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
        f2D = np.tile(f, angular_res).reshape((angular_res, N)).T
    else : 
        f_l = pl_project_2D(f, max_degree, even=False)
        f2D = np.atleast_3d(np.array(pl_eval_2D(f_l, cth_res, der=t_deriv)).T).T[-1]
        
    # Text formating 
    rc('text', usetex=True)
    rc('xtick', labelsize=size)
    rc('ytick', labelsize=size)
    rc('axes', facecolor=background_color)
    
    # Init figure
    norm = None
    if (cmap is cm.Blues)&(np.nanmin(f2D)*np.nanmax(f2D) < -0.01*np.nanmax(np.abs(f2D))**2) : 
        cmap, norm = cm.RdBu_r, mcl.CenteredNorm()
    cbar_width = 0.1
    if add_to_fig is None : 
        margin = 0.05
        x_scale = 2 * margin + (map_res[-1]*sth_res).max() + 4 * cbar_width
        y_scale = 2 * margin + (map_res[-1]*cth_res).max()
        factor = min(18/x_scale, 9.5/y_scale)
        fig, ax = plt.subplots(figsize=(x_scale * factor, y_scale * factor), frameon=False)
        plt.axis('equal')
        plt.xlabel('$s/R_\mathrm{eq}$', fontsize=size+3)
        plt.ylabel('$z/R_\mathrm{eq}$', fontsize=size+3)
        plt.xlim((-1.0, 1.0))
    else : 
        fig, ax = add_to_fig
    
    # Right side
    csr = ax.contourf(
        map_res*sth_res, map_res*cth_res, f2D, 
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
            -map_res*sth_res, map_res*cth_res, f2D, 
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
    fig.tight_layout()
    plt.show()
    
def write_model(fname, params, map_n, additional_var, *args) : 
    """
    Saves the deformed model in the file named fname. The resulting 
    table has dimension (N, M+N_args+N_var) where the last N_var columns
    contains the additional variables given by the user (the lattest
    are left unchanged during the whole deformation). The dimensions N & M,
    as well as the global paramaters mass, radius, ROT, G
    are written on the first line.

    Parameters
    ----------
    fname : string
        File name
    params : tuple
        Model parameters to write on the first line.
    map_n : array_like, shape (N, M)
        level surfaces mapping.
    additional_var : tuple of arrays
        Tuple of (unchanged) additional variables.
    args : tuple with N_args elements
        Variables to be saved in addition to map_n.

    """
    header = "".join(str(c)+" " for c in params)[:-1]
    np.savetxt(
        'Models/'+fname, np.hstack((map_n, np.vstack(args + (*additional_var,)).T)), 
        header=header,  comments=''
    )