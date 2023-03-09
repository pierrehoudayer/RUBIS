import matplotlib.colors as mcl
import matplotlib.pyplot as plt
from matplotlib import rc
from pylab import cm
import numpy as np
from model_deform_sph import (
    find_metric_terms, find_external_mapping
)
from low_level import pl_project_2D, pl_eval_2D, interpolate_func


def mapdiff(map_sph, map_rad, L,
            size=16, n_lev=201, res=501, cmap=cm.seismic) : 
    
    N, M = map_rad.shape
    cth_res = np.linspace(-1, 1, res)
    sth_res = np.sqrt(1-cth_res**2)
    map_l_sph = pl_project_2D(map_sph, L)
    map_l_rad = pl_project_2D(map_rad, L)
    map_res_sph = pl_eval_2D(map_l_sph, cth_res)
    map_res_rad = pl_eval_2D(map_l_rad, cth_res)
    diff = map_res_rad - map_res_sph
    
    plt.close('all')
    plt.contourf(
        map_res_rad*sth_res, map_res_rad*cth_res, diff, 
        cmap=cmap, levels=n_lev, norm=mcl.CenteredNorm()
    )
    plt.contourf(
        -map_res_rad*sth_res, map_res_rad*cth_res, diff, 
        cmap=cmap, levels=n_lev, norm=mcl.CenteredNorm()
    )
    plt.plot( map_res_rad[-1]*sth_res, map_res_rad[-1]*cth_res, 'k--',lw=0.5)
    plt.plot(-map_res_rad[-1]*sth_res, map_res_rad[-1]*cth_res, 'k--',lw=0.5)
    plt.xlabel(r"$s/R_{\mathrm{eq}}$",fontsize=size)
    plt.ylabel(r"$z/R_{\mathrm{eq}}$",fontsize=size)
    cbar = plt.colorbar()
    cbar.set_label(r"$\delta r$", fontsize=size)
    plt.gca().set_aspect("equal")
    plt.show()

def phidiff(phi_l_sph, phi_l_rad, map_sph, r,
            size=16, n_lev=201, res=501, cmap=cm.seismic) : 
    
    N, L = phi_l_rad.shape
    dr_sph = find_metric_terms(map_sph)
    dr_sph = find_external_mapping(dr_sph)
    
    cth_res = np.linspace(-1, 1, res)
    sth_res = np.sqrt(1-cth_res**2)
    map_l_sph = pl_project_2D(dr_sph._, L)
    map_res = pl_eval_2D(map_l_sph, cth_res)
    
    phi2D_sph = pl_eval_2D(phi_l_sph, cth_res)
    phi2D_rad = pl_eval_2D(phi_l_rad, cth_res)
    l = np.arange(L)
    phi2D_int = np.array(
        [np.hstack(
            (interpolate_func(r, phik)(rk[rk < 1]), 
             pl_eval_2D(phi_l_rad[-1] * (rk[rk >= 1, None])**-(l+1), ck))
            )
         for rk, ck, phik in zip(map_res.T, cth_res, phi2D_rad.T)]
        ).T
    diff = phi2D_int - phi2D_sph
    
    plt.close('all')
    plt.contourf(
        map_res*sth_res, map_res*cth_res, diff, 
        cmap=cmap, levels=n_lev, norm=mcl.CenteredNorm()
    )
    plt.contourf(
        -map_res*sth_res, map_res*cth_res, diff, 
        cmap=cmap, levels=n_lev, norm=mcl.CenteredNorm()
    )
    plt.plot( map_res[N-1]*sth_res, map_res[N-1]*cth_res, 'k-',lw=0.5)
    plt.plot(-map_res[N-1]*sth_res, map_res[N-1]*cth_res, 'k-',lw=0.5)
    plt.plot( map_res[-1]*sth_res, map_res[-1]*cth_res, 'k--',lw=0.5)
    plt.plot(-map_res[-1]*sth_res, map_res[-1]*cth_res, 'k--',lw=0.5)
    plt.xlabel(r"$s/R_{\mathrm{eq}}$",fontsize=size)
    plt.ylabel(r"$z/R_{\mathrm{eq}}$",fontsize=size)
    cbar = plt.colorbar()
    cbar.set_label(r"$\delta \phi_g$", fontsize=size)
    plt.gca().set_aspect("equal")
    plt.show()

def phieff_lines(phi_g_l, dphi_g_l, r, zeta, omega, eval_phi_c,
                 size=16, n_lev=51, res=501, cmap=cm.seismic) : 
    
    N, _ = r.shape
    _, L = phi_g_l.shape
    z = zeta[N:].reshape((-1, 1))
        
    surf = r[-1]
    max_degree = 3
    r_ext = np.vstack((r, z - (1-surf)*(2-z)**max_degree))
    
    cth_res = np.linspace(-1, 1, res)
    sth_res = np.sqrt(1-cth_res**2)
    map_l = pl_project_2D(r_ext, L)
    map_res = pl_eval_2D(map_l, cth_res)
    
    phi2D_g = pl_eval_2D(phi_g_l, cth_res)
    phi2D_c = np.array(
        [eval_phi_c(rk, ck, omega)[0] for rk, ck in zip(map_res.T, cth_res)]
    ).T
    phi2D_eff = phi2D_g + phi2D_c
    
    _, dom_unq = np.unique(zeta, return_index=True)
    phi_g_eq  = interpolate_func(
        pl_eval_2D(map_l  , 0.0)[dom_unq], 
        pl_eval_2D(phi_g_l, 0.0)[dom_unq]
    )
    dphi_g_eq = interpolate_func(
        pl_eval_2D(map_l   , 0.0)[dom_unq], 
        pl_eval_2D(dphi_g_l, 0.0)[dom_unq]
    )
    f  = lambda r :  phi_g_eq(r)  + eval_phi_c(r, 0.0, omega)[0]
    df = lambda r : dphi_g_eq(r)  + eval_phi_c(r, 0.0, omega)[1]
    from scipy.optimize import root_scalar
    
    r_crit = root_scalar(df, bracket=[1.0, 2.0]).root
    phi_eff_crit = f(r_crit)
    phi2D_eff -= phi_eff_crit
    
    phimax, phimin = phi2D_eff.max(), phi2D_eff.min()
    levels = (
        list(np.flip(phimin * np.linspace(0.0, 1.0, (n_lev+1)//2) ** 2))[:-1] +
        list(phimax * np.linspace(0.0, 1.0, (n_lev+1)//2) ** 2)
    )
    lw = 0.5 * np.ones(n_lev)
    lw[(n_lev - 1)//2] = 2.0
    if isinstance(cmap, list) :
        cmap = get_continuous_cmap(cmap)
    
    rc('text', usetex=True)
    rc('xtick', labelsize=size)
    rc('ytick', labelsize=size)
    plt.close('all')
    plt.contour(
        map_res*sth_res, map_res*cth_res, phi2D_eff, 
        cmap=cmap, levels=levels, norm=mcl.CenteredNorm(), linewidths=lw
    )
    plt.contour(
        -map_res*sth_res, map_res*cth_res, phi2D_eff, 
        cmap=cmap, levels=levels, norm=mcl.CenteredNorm(), linewidths=lw
    )
    plt.xlabel(r"$s/R_{\mathrm{eq}}$",fontsize=size)
    plt.ylabel(r"$z/R_{\mathrm{eq}}$",fontsize=size)
    plt.gca().set_aspect("equal")
    plt.axis("off")
    plt.show()
    
    
def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]

def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
        
        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcl.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp

def merge_cmaps(cmap1, cmap2) :
    
    def list_to_RGB(color) : 
        return tuple([int(c * 255) for c in color[:-1]])
    
    def RGB_to_hex(color) : 
        return '#%02x%02x%02x' % color
    
    sample = np.linspace(0, 1, 10)
    
    hex_list = (
          [RGB_to_hex(list_to_RGB(c)) for c in cmap1(sample[::-1])] 
        + ['#ffffff']
        + [RGB_to_hex(list_to_RGB(c)) for c in cmap2(sample      )]
        )
    return get_continuous_cmap(hex_list)

cmap = ['#000000', '#000000']