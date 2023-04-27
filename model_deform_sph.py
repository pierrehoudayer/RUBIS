#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 18:36:42 2022

@author: phoudayer
"""

#%% Modules cell

import time
import matplotlib.colors as mcl
import matplotlib.pyplot as plt
import numpy             as np
import scipy.sparse      as sps
import scipy.special     as sp
from matplotlib             import rc
from matplotlib.collections import LineCollection
from pylab                  import cm
from scipy.interpolate      import CubicHermiteSpline
from scipy.linalg.lapack    import dgbsv
from scipy.special          import roots_legendre, eval_legendre

from dotdict                import DotDict
from low_level              import (
    lnxn, 
    del_u_over_v,
    integrate, 
    interpolate_func, 
    pl_eval_2D,
    pl_project_2D,
    Legendre_coupling,
    lagrange_matrix_P, 
    app_list, 
    plot_f_map
)
from rotation_profiles      import solid, lorentzian, plateau, la_bidouille 
from generate_polytrope     import polytrope, composite_polytrope  


#%% High-level functions cell
    
def set_params() : 
    """
    Function returning the main parameters of the file.

    Returns
    -------
    model_choice : string or DotDict instance
        Name of the file containing the 1D model or dictionary containing
        the information requiered to compute a composite polytrope with
        caracteristics: {
            indices : array_like, shape(D, )
                Each region polytropic index
            target_pressures : array_like, shape(D, )
                Normalised interface pressure values (surface included).
            density_jumps : array_like, shape(D-1, )
                Density ratios above and below each interface (surface excluded).
                The default value is None
            R : float, optional
                Composite polytrope radius. The default value is 1.0
            M : float, optional
                Composite polytrope mass. The default value is 1.0
            res : int, optional
                Number of points. The default value is 1001
        }
        Please refer to the composite_polytrope() documentation for more information.
    rotation_profile : function(r, cth, omega)
        Function used to compute the centrifugal potential and its 
        derivative. Possible choices are {solid, lorentzian, plateau}.
        Explanations regarding this profiles are available in the 
        corresponding functions.
    rotation_target : float
        Target for the rotation rate.
    rate_difference : float
        The rotation rate difference between the centre and equator in the
        cylindrical rotation profile. For instance, rate_difference = 0.0
        would corresond to a solid rotation profile while 
        rate_difference = 0.5 indicates that the star's centre rotates 50%
        faster than the equator.
        Only appears in cylindrical rotation profiles.
    rotation_scale : float
        Homotesy factor on the x = r*sth / Req axis for the rotation profile.
        Only appear in the plateau rotation profile.
    max_degree : integer
        Maximum l degree to be considered in order to do the
        harmonic projection.
    angular_resolution : integer
        Angular resolution for the mapping. Better to take an odd number 
        in order to include the equatorial radius.
    full_rate : integer
        Number of iterations before reaching full rotation rate. As a 
        rule of thumb, ~ 1 + int(-np.log(1-rotation_target)) iterations
        should be enough to ensure convergence (in the solid rotation case!).
    mapping_precision : float
        Precision target for the convergence criterion on the mapping.
    newton_precision : float
        Precision target for Newton's method.
    lagrange_order : integer
        Choice of Lagrange polynomial order in integration / interpolation
        routines. 
        2 should be enough.
    spline_order : integer
        Choice of B-spline order in integration / interpolation
        routines. 
        3 is recommanded (must be odd in anycase).
    save_resolution : integer
        Angular resolution for saving the mapping.
    save_name : string
        Filename in which to scaled model will be saved.
        
    external_domain_res : integer
        Radial resolution for the external domain
    derivable_mapping : boolean
        Allows to choose between two external mapping prescriptions
        (cf. find_external_mapping routine).

    """
    
    #### MODEL CHOICE ####
    # model_choice = "Jupiter.txt"   
    model_choice = DotDict(index=3.0, surface_pressure=0.0, R=1.0, M=1.0, res=1000)
    model_choice = DotDict(
        indices = (1.0, 2.0, 3.0), 
        target_pressures = (-1.0, -3.0, -np.inf), 
        density_jumps = (0.5, 0.5),
        R=1.0, M=1.0, res=1001
    )

    #### ROTATION PARAMETERS ####      
    rotation_profile = solid
    rotation_target = 0.8
    central_diff_rate = 2.0
    rotation_scale = 1.0
    
    #### SOLVER PARAMETERS ####
    max_degree = angular_resolution = 101
    full_rate = 5
    mapping_precision = 1e-10
    newton_precision = 1e-11
    lagrange_order = 2
    spline_order = 3
    
    #### OUTPUT PARAMETERS ####
    plot_resolution = 501
    save_name = give_me_a_name(model_choice, rotation_target)
    
    #### EXT. MAPPING PARAMETERS ####
    external_domain_res = 1001
    derivable_mapping = False
    
    return (
        model_choice, rotation_target, full_rate, rotation_profile,
        central_diff_rate, rotation_scale, mapping_precision,
        newton_precision, spline_order, lagrange_order, max_degree, 
        angular_resolution, plot_resolution, save_name,
        external_domain_res, derivable_mapping
    )

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

def init_1D() : 
    """
    Function reading the 1D model file 'MOD_1D' (or generating a
    polytrope if MOD_1D is a dictionary). If additional variables are 
    found in the file, they are left unchanged and returned in the 
    'SAVE' file.

    Returns
    -------
    P0 : float
        Value of the surface pressure after normalisation.
    N : integer
        Radial resolution of the model.
    M : float
        Total mass of the model.
    R : float
        Radius of the model.
    r : array_like, shape (N, ), [GLOBAL VARIABLE]
        Radial coordinate after normalisation.
    zeta : array_like, shape (N+Ne, ), [GLOBAL VARIABLE]
        Spheroidal coordinate
    rho : array_like, shape (N, )
        Radial density of the model after normalisation.
    other_var : array_like, shape (N, N_var)
        Additional variables found in 'MOD1D'.

    """
    if isinstance(MOD_1D, DotDict) :        
        # The model properties are user-defined
        N = MOD_1D.res    or 1001
        M = MOD_1D.mass   or 1.0
        R = MOD_1D.radius or 1.0
        
        # Polytrope computation
        # model = polytrope(*MOD_1D.values())
        model = composite_polytrope(*MOD_1D.values())
        
        # Normalisation
        r   = model.r     /  R
        rho = model.rho   / (M/R**3)
        other_var = np.empty_like(r)
        P0  = model.p[-1] / (G*M**2/R**4)
        
    else : 
        # Reading file 
        surface_pressure, radial_res = np.genfromtxt(
            './Models/'+MOD_1D, max_rows=2, unpack=True
        )
        N = int(radial_res)
        r1D, rho1D, *other_var = np.genfromtxt(
            './Models/'+MOD_1D, skip_header=2, unpack=True
        )
        _, idx = np.unique(r1D, return_index=True) 
        
        # Normalisation
        R = r1D[-1]
        M = 4*np.pi * integrate(x=r1D[idx], y=r1D[idx]**2 * rho1D[idx])
        r   = r1D / (R)
        rho = rho1D / (M/R**3)
        P0  = surface_pressure / (M**2/R**4)
        
        # We assume that P0 is already normalised by G if R ~ 1 ...
        if not np.allclose(R, 1) :
            P0 /= G
    
    # Spheroidal coordinate
    re = np.linspace(1, 2, Ne)
    zeta = np.hstack((r, 1+sp.betainc(2, 2, re-1)))
    
    return P0, N, M, R, r, zeta, rho, other_var
        
def init_2D() :
    """
    Init function for the angular domain.

    Parameters
    ----------
    None.

    Returns
    -------
    cth : array_like, shape (M, ), [GLOBAL VARIABLE]
        Angular coordinate (equivalent to cos(theta)). 
    weights : array_like, shape (M, ), [GLOBAL VARIABLE]
        Angular weights for the Legendre quadrature.
    map_n : array_like, shape (N, M)
        Isopotential mapping 
        (given by r(phi_eff, theta) = r for now).
    """
    cth, weights = roots_legendre(M)
    map_n = np.tile(r, (M, 1)).T
    return cth, weights, map_n   

def init_phi_c() : 
    """
    Defines the function used to compute the centrifugal potential
    with the adequate arguments.

    Returns
    -------
    phi_c : function(r, cth, omega)
        Centrifugal potential

    """
    nb_args = PROFILE.__code__.co_argcount - len(PROFILE.__defaults__ or '')
    mask = np.array([0, 1]) < nb_args - 3
    args = np.array([ALPHA, SCALE])[mask]
    phi_c = lambda r, cth, omega : PROFILE(r, cth, omega, *args)
    return phi_c

def find_r_eq(map_n) :
    """
    Function to find the equatorial radius from the mapping.

    Parameters
    ----------
    map_n : array_like, shape (N, M)
        Isopotential mapping.

    Returns
    -------
    r_eq : float
        Equatorial radius.

    """
    surf_l = pl_project_2D(map_n[-1], L)
    return pl_eval_2D(surf_l, 0.0)

def find_r_pol(map_n) :
    """
    Function to find the polar radius from the mapping.

    Parameters
    ----------
    map_n : array_like, shape (N, M)
        Isopotential mapping.

    Returns
    -------
    r_eq : float
        Equatorial radius.

    """
    surf_l = pl_project_2D(map_n[-1], L)
    return pl_eval_2D(surf_l, 1.0)
    

def find_mass(map_n, rho_n) :
    """
    Find the total mass over the mapping map_n.

    Parameters
    ----------
    map_n : array_like, shape (N, M)
        Isopotential mapping.
    rho_n : array_like, shape (N, )
        Density profile (the same in each direction).
        
    Returns
    -------
    mass_tot : float
        Total mass integrated of map_n.

    """
    # Starting by the computation of the mass in each angular direction
    mass_ang = np.array([sum(
        integrate(rk[D], rho_n[D] * rk[D]**2, k=5) for D in dom.ranges[:-1]
    ) for rk in map_n.T])
    
    # Integration of the angular domain
    mass_tot = 2*np.pi * mass_ang @ weights
    return mass_tot


def find_gravitational_moments(map_n, rho_n, max_degree=14) :
    """
    Find the gravitational moments up to max_degree.

    Parameters
    ----------
    map_n : array_like, shape (N, M)
        Isopotential mapping.
    rho_n : array_like, shape (N, )
        Density profile (the same in each direction).
    max_degree : int, optional
        Maximum degree for the gravitational moments. The default is 10.
        
    Returns
    -------
    moments : array_like, shape (max_degree//2 + 1, )
        All the gravitational moments.

    """
    # Degrees list
    degrees = np.arange(max_degree+1, step=2)
    
    # Moments integration in each direction
    MltD = lambda l, r, D : integrate(r[D], rho_n[D] * r[D]**(l+2), k=5)
    moments_ang = - np.array(
        [[sum(MltD(l, rk, D) for D in dom.ranges[:-1]) for rk in map_n.T] for l in degrees]
    )
    pl = np.array([eval_legendre(l, cth) for l in degrees])                                                      
    
    # Integration over the angular domain
    moments = 2*np.pi * (moments_ang * pl) @ weights  
    
    return moments 


def find_pressure(rho, dphi_eff) :
    """
    Find the pressure evaluated on zeta thanks to the hydrostatic
    equilibrium.

    Parameters
    ----------
    rho : array_like, shape (N, )
        Density profile.
    dphi_eff : array_like, shape (N, )
        Effective potential derivative with respect to zeta.

    Returns
    -------
    P : array_like, shape (N, )
        Pressure profile.

    """
    dP = - rho * dphi_eff[dom.int]        
    P  = interpolate_func(
        zeta[dom.unq_int], dP[dom.unq_int], der=-1, k=KSPL, prim_cond=(-1, P0)
    )(zeta[dom.int])
    return P


def find_phi_eff(map_n, rho_n, phi_eff=None) :
    """
    Determination of the effective potential from a given mapping
    (map_n, which gives the lines of constant density), and a given 
    rotation rate (omega_n). This potential is determined by solving
    the Poisson's equation on each degree of the harmonic decomposition
    (giving the gravitational potential harmonics which are also
    returned) and then adding the centrifugal potential.

    Parameters
    ----------
    map_n : array_like, shape (N, M)
        Current mapping.
    rho_n : array_like, shape (N, )
        Current density on each equipotential.
    phi_eff : array_like, shape (N, ), optional
        If given, the current effective potential on each 
        equipotential. If not given, it will be calculated inside
        this fonction. The default is None.

    Raises
    ------
    ValueError
        If the matrix inversion enconters a difficulty ...

    Returns
    -------
    phi_g_l : array_like, shape (N, L), optional
        Gravitation potential harmonics.
    phi_eff : array_like, shape (N, )
        Effective potential on each equipotential.
    dphi_eff : array_like, shape (N, ), optional
        Effective potential derivative with respect to zeta.

    """    
    # Empty harmonics initialisation
    phi_g_l  = np.zeros((N+Ne, L))
    dphi_g_l = np.zeros((N+Ne, L))
    
    # Metric terms and coupling integral computation
    dr     = find_metric_terms(map_n)
    r2rz_l = pl_project_2D(dr._**2 * dr.z, L) / (np.arange(L)+0.5)
    dr     = find_external_mapping(dr)
    Pll    = find_all_couplings(dr)
    
    # Vector and band matrix storage
    Nl = (L+1)//2
    l  = np.arange(0, L, 2)
    kl = (2*KLAG + 1) * Nl - 1
    ku = (2*KLAG + 1) * Nl - 1
    b  = np.zeros((2*(N+Ne)*Nl, ))
    blocs = np.empty((kl+ku-L+1, 2*(N+Ne)*Nl))
    ab = np.zeros((2*kl+ku+1, 2*(N+Ne)*Nl))
    
    for d, D in enumerate(dom.ranges) : 
        
        # Domain properties
        beg_i, end_i = (2*dom.edges[d]+1)*Nl, (2*dom.edges[d+1]-1)*Nl
        beg_j, end_j = (2*dom.edges[d]+0)*Nl, (2*dom.edges[d+1]-0)*Nl
        Lsp_d_broad = Lsp[d].data[::-1, :, None, None]
        Dsp_d_broad = Dsp[d].data[::-1, :, None, None]
        size = dom.sizes[d]
        
        # Vector filling
        if d < dom.Nd - 1 : 
            b[beg_i:end_i:2] = 4*np.pi * (
                Lsp[d] @ (rho_n[D, None] * r2rz_l[D, ::2])
                ).reshape((-1))
            
        # Main matrix parts filling
        temp = np.empty((2*KLAG, size, 2*Nl, 2*Nl))
        temp[..., 0::2, 0::2] = (
            + Dsp_d_broad * Pll.zz[D]
            - Lsp_d_broad * Pll.zt[D]
            )
        temp[..., 0::2, 1::2] = - Lsp_d_broad * Pll.tt[D]
        temp[..., 1::2, 0::2] = + Lsp_d_broad * np.eye(Nl)
        temp[..., 1::2, 1::2] = - Dsp_d_broad * np.eye(Nl)
        blocs[:, beg_j:end_j] = np.moveaxis(temp, 2, 1).reshape(
            (2*KLAG*2*Nl, 2*size*Nl)
        )
            
        # Inner boundary conditions 
        if d == 0 :
            blocs[ku-2*Nl+1:ku-Nl+1, 0:2*Nl:2] = np.diag((1, ) + (0, )*(Nl-1))
            blocs[ku-2*Nl+1:ku-Nl+1, 1:2*Nl:2] = np.diag((0, ) + (1, )*(Nl-1))
        else :  
            blocs[ku-3*Nl+1+0:ku-Nl+1:2, beg_j+0:beg_j+2*Nl:2] = -Pll.BC[D[0]]
            blocs[ku-3*Nl+1+1:ku-Nl+1:2, beg_j+1:beg_j+2*Nl:2] = -np.eye(Nl)
        
        # Outer boundary conditions
        if d == dom.Nd - 1 : 
            blocs[ku-Nl+1:ku+1, -2*Nl+0::2] = np.eye(Nl)
            blocs[ku-Nl+1:ku+1, -2*Nl+1::2] = np.diag((l+1)/2)
        else :  
            blocs[ku-Nl+1+0:ku+Nl+1:2, end_j-2*Nl+0:end_j:2] = Pll.BC[D[-1]]
            blocs[ku-Nl+1+1:ku+Nl+1:2, end_j-2*Nl+1:end_j:2] = np.eye(Nl)
    
    # Matrix reindexing (credits to N. Fargette for this part)
    mask = np.zeros((2*(N+Ne)*Nl, kl+ku+1), dtype=bool)
    for l in range(2*Nl) : 
        mask[l::2*Nl, L-l:kl+ku+1-l] = 1
    (ab[kl:, :]).T[mask] = (blocs.T).flatten()
                
    # Matrix inversion (LAPACK)
    _, _, x, info = dgbsv(kl, ku, ab, b)

    if info != 0 : 
        raise ValueError(
            "Problem with finding the gravitational potential. \n",
            "Info = ", info
            )
            
    # Poisson's equation solution
    phi_g_l[: , ::2] = x[1::2].reshape((N+Ne, Nl))
    dphi_g_l[:, ::2] = x[0::2].reshape((N+Ne, Nl))    
    
    if phi_eff is None :

        # First estimate of the effective potential and its derivative
        phi_eff  = pl_eval_2D( phi_g_l, 0.0)
        dphi_eff = pl_eval_2D(dphi_g_l, 0.0)        
        return phi_g_l, dphi_g_l, phi_eff, dphi_eff
        
    # The effective potential is known to an additive constant 
    C = pl_eval_2D(phi_g_l[0], 0.0) - phi_eff[0]
    phi_eff += C
    return phi_g_l, dphi_g_l, phi_eff
    

def find_centrifugal_potential(r, cth, omega, dim=False) :
    """
    Determination of the centrifugal potential and its 
    derivative in the case of a cylindric rotation profile 
    (caracterised by ALPHA). The option dim = True allows a
    computation taking into account the future renormalisation
    (in this case r_eq != 1 but r_eq = R_eq / R).

    Parameters
    ----------
    r : float or array_like, shape (Nr, )
        Radial value(s).
    cth : float
        Value of cos(theta).
    omega : float
        Rotation rate.
    dim : boolean, optional
        Set to true for the omega computation. 
        The default is False.

    Returns
    -------
    phi_c : float or array_like, shape (Nr, )
        Centrifugal potential.
    dphi_c : float or array_like, shape (Nr, )
        Centrifugal potential derivative with respect to r.

    """
    phi_c, dphi_c = eval_phi_c(r, cth, omega)
    if dim :
        return phi_c / r**3, (r*dphi_c - 3*phi_c) / r**4
    return phi_c, dphi_c


def estimate_omega(phi_g, target, omega_n) :
    """
    Estimates the adequate rotation rate so that it reaches ROT
    after normalisation. Considerably speed-up (and stabilises)
    the overall convergence.

    Parameters
    ----------
    phi_g : function(r_eval)
        Gravitational potential along the equatorial cut.
    target : float
        Value to reach for the effective potential.
    omega_n : float
        Current rotation rate.

    Returns
    -------
    omega_n_new : float
        New rotation rate.

    """    
    # Searching for a new omega
    r_est  = 1.0
    del_r  = 1.0
    while abs(del_r) > DELTA : 
        
        # Gravitational potential
        phi_g_est  = phi_g(r_est, nu=0)
        dphi_g_est = phi_g(r_est, nu=1)
        
        # Centrifugal potential 
        phi_c_est, dphi_c_est = find_centrifugal_potential(
            r_est, 0.0, omega_n, dim=True 
        )
        
        # Total potential
        phi_t_est  =  phi_g_est +  phi_c_est
        dphi_t_est = dphi_g_est + dphi_c_est
        
        # Update r_est
        del_r = (target - phi_t_est) / dphi_t_est
        r_est += del_r
        assert r_est < 2    
    
    # Updating omega
    omega_n_new = omega_n * r_est**(-1.5)
    return omega_n_new


def find_new_mapping(map_n, omega_n, phi_g_l, dphi_g_l, phi_eff) :
    """
    Find the new mapping by comparing the effective potential
    and the total potential (calculated from phi_g_l and omega_n).

    Parameters
    ----------
    map_n : array_like, shape (N, M)
        Current mapping.
    omega_n : float
        Current rotation rate.
    phi_g_l : array_like, shape (N, L)
        Gravitation potential harmonics.
    dphi_g_l : array_like, shape (N, L)
        Gravitation potential derivative harmonics.
    phi_eff : array_like, shape (N, )
        Effective potential on each equipotential.

    Returns
    -------
    map_n_new : array_like, shape (N, M)
        Updated mapping.
    omega_n_new : float
        Updated rotation rate.

    """
    # Find metric terms
    dr = find_metric_terms(map_n)
    dr = find_external_mapping(dr)
    
    # 2D gravitational potential
    phi2D_g  = pl_eval_2D( phi_g_l, cth)
    dphi2D_g = pl_eval_2D(dphi_g_l, cth) / dr.z
        
    # Gravitational potential interpolation
    up = np.arange((M+1)//2)
    phi_g_func  = [CubicHermiteSpline(
        x=dr._[dom.unq, k], y=phi2D_g[dom.unq, k], dydx=dphi2D_g[dom.unq, k]
    ) for k in up]
    
    # Find a new value for ROT
    target = phi_eff[N-1] - pl_eval_2D(phi_g_l[0], 0.0) + phi_eff[0]
    omega_n_new = estimate_omega(phi_g_func[-1], target, omega_n)
    
    # Find the mapping
    map_est = np.copy(map_n[:, up])
    idx = np.indices(map_est.shape)
    del_r = np.ones_like(map_n[:, up])
    del_r[0] = 0.0        
    
    while np.any(np.abs(del_r) > DELTA) :
        
        # Terms that require corrections
        C_est = (np.abs(del_r) > DELTA)
        r_est = map_est[C_est]
        k_est = idx[1,  C_est]        
            
        # Gravitational potential
        inv_sort = np.argsort(app_list(np.arange(len(k_est)), k_est))
        phi_g_est = np.array(
            (app_list(r_est, k_est, phi_g_func        ),
             app_list(r_est, k_est, phi_g_func, (1,)*M))
        )[:, inv_sort]
        
        # Centrifugal potential
        phi_c_est = np.array(
            find_centrifugal_potential(r_est, cth[k_est], omega_n_new)
        )
        
        # Total potential
        phi_t_est =  phi_g_est +  phi_c_est
        
        # Update map_est
        del_r[C_est] = (phi_eff[idx[0, C_est]] - phi_t_est[0]) / phi_t_est[1]
        map_est[C_est] += del_r[C_est]
    
    # New mapping
    map_n_new = np.hstack((map_est, np.flip(map_est, axis=1)[:, 1:]))
        
    return map_n_new, omega_n_new

    
def write_model(fname, map_n, *args) : 
    """
    Saves the deformed model in the file named fname. The resulting 
    table has dimension (N, M+N_args+N_var) where the last N_var columns
    contains the additional variables given by the user (the lattest
    are left unchanged during the whole deformation). The dimensions N & M,
    as well as the global paramaeters mass, radius, ROT, G
    are written on the first line.

    Parameters
    ----------
    fname : string
        File name
    map_n : array_like, shape (N, M)
        level surfaces mapping.
    args : TUPLE with N_args elements
        Variables to be saved in addition to map_n.

    """
    np.savetxt(
        'Models/'+fname, np.hstack((map_n, np.vstack(args + (*VAR,)).T)), 
        header=f"{N} {M} {mass} {radius} {ROT} {G}", 
        comments=''
    )
        
        
    
    
#%% Sph functions cell

def find_domains() :
    """
    Defines many tools to help the domain manipulation and navigation.

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
            id : array_like, shape (N+Ne, ) of integer
                Domain identification number. 
            id_val : array_like, shape (Nd, ) of integer
                The id values.
            int, ext : array_like, shape (N+Ne, ) of boolean
                Interior (resp. exterior, i.e. if rho = 0) domain.
            unq : array_like, shape (N+Ne-(Nd-1), ) of integer
                Unique indices through the domains.
            }

    """
    dom = DotDict()
    
    # Domain physical boundaries
    unq, unq_idx, unq_inv, unq_cnt = np.unique(
        zeta, return_index=True, return_inverse=True, return_counts=True
    )
    cnt_mask = unq_cnt > 1
    dom.bounds = unq[cnt_mask]
    
    # Domain interface indices
    cnt_idx, = np.nonzero(cnt_mask)
    idx_mask = np.in1d(unq_inv, cnt_idx)
    idx_idx, = np.nonzero(idx_mask)
    srt_idx  = np.argsort(unq_inv[idx_mask])
    dom.interfaces = np.split(
        idx_idx[srt_idx], np.cumsum(unq_cnt[cnt_mask])[:-1]
    )
    dom.end, dom.beg = np.array(dom.interfaces).T
    
    # Domain ranges and sizes
    dom.unq    = unq_idx
    dom.Nd     = len(dom.bounds) + 1
    dom.edges  = np.array((0, ) + tuple(dom.beg) + (N+Ne, ))
    dom.ranges = list(map(range, dom.edges[:-1], dom.edges[1:]))
    dom.sizes  = list(map(len, dom.ranges))
    
    # Domain indentification
    dom.id      = np.hstack([d*np.ones(S) for d, S in enumerate(dom.sizes)])
    dom.id_val  = np.unique(dom.id)
    dom.ext     = dom.id == dom.Nd - 1
    dom.int     = np.invert(dom.ext)
    dom.unq_int = np.unique(zeta[dom.int], return_index=True)[1]
    
    return dom

def find_sparse_matrices_per_domain() : 
    """
    Finds the interpolation and derivation matrices (in scipy sparse
    storage format) for each spheroidal domain.

    Returns
    -------
    Lsp, Dsp : list of array_like, shape (size_domain - 1, size_domain)
        Interpolation and derivation matrices.

    """
    Lsp, Dsp = [], []
    for D in dom.ranges : 
        
        # Find the lagrange matrices per domain
        lag_mat = lagrange_matrix_P(zeta[D], order=KLAG)
        
        # Define sparse matrices 
        Lsp.append(sps.dia_matrix(lag_mat[..., 0]))
        Dsp.append(sps.dia_matrix(lag_mat[..., 1]))
    
    return Lsp, Dsp
    

def find_metric_terms(map_n) : 
    """
    Finds the metric terms, i.e the derivatives of r(z, t) 
    with respect to z or t (with z := zeta and t := cos(theta)).

    Parameters
    ----------
    map_n : array_like, shape (N, M)
        Isopotential mapping.

    Returns
    -------
    dr : DotDict instance
        The mapping derivatives : {
            _   = r(z, t),
            t   = r_t(z, t),
            tt  = r_tt(z, t),
            z   = r_z(z, t),
            zt  = r_zt(z, t),
            ztt = r_ztt(z, t)
            }          
    """
    dr = DotDict()
    dr._ = map_n
    map_l = pl_project_2D(dr._, L)
    _, dr.t, dr.tt = pl_eval_2D(map_l, cth, der=2)
    dr.z = np.array(
        [np.hstack(
            [interpolate_func(zeta[D], rk[D], der=1, k=KSPL)(zeta[D]) 
             for D in dom.ranges[:-1]]
        ) for rk in map_n.T]            # <- mapping derivative potentially
    ).T                                 #    discontinous on the interfaces
    map_l_z = pl_project_2D(dr.z, L)
    _, dr.zt, dr.ztt = pl_eval_2D(map_l_z, cth, der=2)
    return dr

def find_external_mapping(dr) : 
    """
    Complete the internal mapping (0 <= z <= 1) to an external 
    domain (1 <= z <= 2). By convention, this continuation reaches 
    the spherical metric at z=2, thus making it handy to impose 
    the boundary conditions on this point. The two options proposed
    consist in computing a mapping preserving dr/dz through the surface
    (DERIV = True) or not (DERIV = False). The latter is recommended.

    Parameters
    ----------
    dr : DotDict instance
        The internal mapping and its derivatives with respect to z and t

    Returns
    -------
    dr : DotDict instance
        The commplete mapping from 0 <= z <= 2.
    """
    
    if DERIV : 
        ### DERIVABLE MAPPING ####
        
        # External zeta variable 
        z = zeta[dom.ext].reshape((-1, 1))
        
        # a's derivatives
        da_u = (1 + dr.z[-1], dr.zt[-1], dr.ztt[-1])
        da_v = (2 - dr._[-1], -dr.t[-1], -dr.tt[-1])
        a   = del_u_over_v(da_u, da_v, 0)
        da  = del_u_over_v(da_u, da_v, 1)
        dda = del_u_over_v(da_u, da_v, 2)
        
        # b's derivatives
        db_u = (dr._[-1]  + 2*dr.z[-1], 
                dr.t[-1]  + 2*dr.zt[-1], 
                dr.tt[-1] + 2*dr.ztt[-1])
        db_v = (1 + dr.z[-1], dr.zt[-1], dr.ztt[-1])
        b   = del_u_over_v(db_u, db_v, 0)
        db  = del_u_over_v(db_u, db_v, 1)
        ddb = del_u_over_v(db_u, db_v, 2)
        
        # r's derivatives
        dr_u = (
            (z-1)**a - dr.z[-1]*(2-z)**a, 
            da*lnxn(z-1, 1, a) - dr.z[-1]*da*lnxn(2-z, 1, a)
              - dr.zt[-1]*(2-z)**a,
            da**2*lnxn(z-1, 2, a) - dr.z[-1]*da**2*lnxn(2-z, 2, a)
              + dda*lnxn(z-1, 1, a) - dr.z[-1]*dda*lnxn(2-z, 1, a)
              - 2*dr.zt[-1]*da*lnxn(2-z, 1, a) - dr.ztt[-1]*(2-z)**a
        )
        dr_v = (a, da, dda)
        
        # External mapping and derivatives
        dr._  = np.vstack((dr._,  del_u_over_v(dr_u, dr_v, 0) + b)     )
        dr.t  = np.vstack((dr.t,  del_u_over_v(dr_u, dr_v, 1) + db)    )
        dr.tt = np.vstack((dr.tt, del_u_over_v(dr_u, dr_v, 2) + ddb)   )
        dr.z  = np.vstack((dr.z, (z-1)**(a-1) + dr.z[-1]*(2-z)**(a-1)) )
    
    else :
        #### NON DERIVABLE MAPPING ####
        
        # External zeta variable 
        z = zeta[dom.ext].reshape((-1, 1))
        
        # Internal mapping constraints
        surf, dsurf, ddsurf = dr._[-1], dr.t[-1], dr.tt[-1]
        
        # External mapping and derivatives
        max_degree = 3
        dr._  = np.vstack((dr._, z - (1-surf)*(2-z)**max_degree))
        dr.t  = np.vstack((dr.t,       dsurf *(2-z)**max_degree))
        dr.tt = np.vstack((dr.tt,     ddsurf *(2-z)**max_degree))
        dr.z  = np.vstack((dr.z, 1 + (1-surf)*(2-z)**(max_degree-1)*max_degree))
    
    
    # Sanity check
    assert not np.any(dr.z < 0)
    
    return dr

def find_all_couplings(dr, alpha=2) :
    """
    Find all the couplings needed to solve Poisson's equation in 
    spheroidal coordinates.

    Parameters
    ----------
    dr : DotDict instance
        The mapping and its derivatives with respect to z and t
    alpha : float, optional
        Constant to be either set to 1 (typically for divergences)
        or 2 (Laplacians).

    Returns
    -------
    Pll : DotDict instance
        Harmonic couplings. They are caracterised by their related 
        metric term : {
            zz : array_like, shape (N+Ne, Nl, Nl)
                coupling associated to phi_zz,
            zt : array_like, shape (N+Ne, Nl, Nl)
                            //         phi_zt,
            tt : array_like, shape (N+Ne, Nl, Nl)
                            //         phi_tt,
            BC : array_like, shape (N+Ne, Nl, Nl)
                coupling used to ensure the gradient continuity.
        }

    """
    Pll = DotDict()
    l = np.arange(0, L, 2)
    
    Pll.zz = Legendre_coupling(
        (dr._**2 + (1-cth**2) * dr.t**2) / dr.z, L, der=(0, 0)
    )
    Pll.zt = Legendre_coupling(
        (1-cth**2) * dr.tt - 2*cth * dr.t, L, der=(0, 0)
    ) + alpha * Legendre_coupling(
        (1-cth**2) * dr.t, L, der=(0, 1)
    )
    Pll.tt = Legendre_coupling(dr.z, L, der=(0, 0)) * l*(l+1)
    
    Pll.BC = Legendre_coupling(1/dr.z, L, der=(0, 0))
    
    return Pll    
    

#%% Main cell

if __name__ == '__main__' :
    
    start = time.perf_counter()
    
    # Definition of global parameters
    MOD_1D, ROT, FULL, PROFILE, ALPHA, SCALE, EPS, DELTA, KSPL, \
    KLAG, L, M, RES, SAVE, Ne, DERIV = set_params()
    G = 6.6743e-11     # <- value of the gravitational constant
    
    # Definition of the 1D-model
    P0, N, mass, radius, r, zeta, rho_n, VAR = init_1D() 
    
    # Domains identification
    dom = find_domains()
    
    # Angular domain preparation
    cth, weights, map_n = init_2D()
    
    # Centrifugal potential definition
    eval_phi_c = init_phi_c()
    
    # Find the lagrange matrices per domain
    Lsp, Dsp = find_sparse_matrices_per_domain()
    
    # Initialisation for the effective potential
    phi_g_l, dphi_g_l, phi_eff, dphi_eff = find_phi_eff(map_n, rho_n)
    
    # Find pressure
    P = find_pressure(rho_n, dphi_eff)
    
    # Iterative centrifugal deformation
    surfaces = [map_n[-1]]
    r_pol = [0.0, find_r_pol(map_n)]
    n = 1
    
    # SAVE
    phi_g_l_sph = [phi_g_l]
    dphi_g_l_sph = [dphi_g_l]
    phi_eff_sph = [np.copy(phi_eff)]
    map_n_sph = [map_n]
    omega_n_sph = [0.0]
    
    # while n < 11 :
    while abs(r_pol[-1] - r_pol[-2]) > EPS :
        
        # Current rotation rate
        omega_n = min(ROT, (n/FULL) * ROT)
        
        # Effective potential computation
        phi_g_l, dphi_g_l, phi_eff = find_phi_eff(map_n, rho_n, phi_eff)
        
        # SAVE
        phi_g_l_sph.append(phi_g_l)
        dphi_g_l_sph.append(dphi_g_l)
        phi_eff_sph.append(np.copy(phi_eff))
        
        # Update the mapping
        map_n, omega_n = find_new_mapping(
            map_n, omega_n, phi_g_l, dphi_g_l, phi_eff
        )        

        # Renormalisation
        r_corr    = find_r_eq(map_n)
        m_corr    = find_mass(map_n, rho_n)   
        radius   *= r_corr
        mass     *= m_corr
        map_n    /=             r_corr
        rho_n    /= m_corr    / r_corr**3
        phi_eff  /= m_corr    / r_corr
        dphi_eff /= m_corr    / r_corr    # <- this a derivative w.r.t. to zeta
        P        /= m_corr**2 / r_corr**4
        
        # Update the surface and polar radius
        surfaces.append(map_n[-1])
        r_pol.append(find_r_pol(map_n))
        
        # SAVE
        map_n_sph.append(map_n)
        omega_n_sph.append(omega_n)
        
        # Iteration count
        DEC = int(-np.log10(EPS))
        print(f"Iteration n°{n}, R_pol = {r_pol[-1].round(12)},")
        n += 1
        
    finish = time.perf_counter()
    print(f'Deformation done in {round(finish-start, 4)} sec')
    
    # Plot mapping
    dr = find_metric_terms(map_n)
    dr = find_external_mapping(dr)
    plot_f_map(
        map_n, np.log10(rho_n+np.max(rho_n)*1e-5)+16, phi_eff, L, map_ext=dr._[N:],
        cmap=cm.viridis_r, disc=dom.end[:-1], n_lines_ext=15
    )
    
    # # Model scaling
    # map_n    *=               radius
    # rho_n    *=     mass    / radius**3
    # phi_eff  *= G * mass    / radius   
    # dphi_eff *= G * mass    / radius**2
    # P        *= G * mass**2 / radius**4
    
    # Model writing
    # write_model(SAVE, map_n, P, rho_n, phi_eff[:N])
    
        
    
    
    
    

    