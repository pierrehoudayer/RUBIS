#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 18:36:42 2022

@author: phoudayer
"""

#%% Modules cell

import time
import matplotlib.pyplot as plt
import numpy             as np
import scipy.sparse      as sps
from matplotlib             import rc
from matplotlib.collections import LineCollection
from pylab                  import cm 
from scipy.interpolate      import CubicHermiteSpline
from scipy.linalg.lapack    import dgbsv
from scipy.special          import roots_legendre, eval_legendre

from dotdict                import DotDict
from low_level              import (
    integrate, 
    interpolate_func, 
    pl_eval_2D,
    pl_project_2D,
    lagrange_matrix_P,
    app_list,
    plot_f_map
)
from rotation_profiles      import solid, lorentzian, plateau, la_bidouille 
from generate_polytrope     import polytrope

#%% High-level functions cell
    
def set_params() : 
    """
    Function returning the main parameters of the file.
    
    Returns
    -------
    model_choice : string or DotDict instance
        Name of the file containing the 1D model or dictionary containing
        the information requiered to compute a polytrope of given
        index : {
            index : float
                Polytrope index
            surface_pressure : float
                Surface pressure expressed in units of central
                pressure, ex: 1e-12 => P0 = 1e-12 * PC
            radius : float
                Radius of the model
            mass : float
                Mass of the model
            res : integer
                Radial resolution of the model
        }
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
    plot_resolution : integer
        Angular resolution for the mapping plot.
    save_name : string
        Filename in which to scaled model will be saved.

    """
    #### MODEL CHOICE ####
    # model_choice = "1Dmodel_1.97187607_G1.txt"     
    model_choice = DotDict(index=1.0, surface_pressure=0.0, R=1.0, M=1.0, res=1000)

    #### ROTATION PARAMETERS ####      
    rotation_profile = solid
    # rotation_profile = la_bidouille('rota_eq.txt', smoothing=1e-5)
    rotation_target = 0.089195487 ** 0.5
    central_diff_rate = 0.5
    rotation_scale = 1.0
    
    #### SOLVER PARAMETERS ####
    max_degree = angular_resolution = 101
    full_rate = 1
    mapping_precision = 1e-10
    newton_precision = 1e-11
    lagrange_order = 2
    spline_order = 3
    
    #### OUTPUT PARAMETERS ####
    plot_resolution = 501
    save_name = give_me_a_name(model_choice, rotation_target)
    
    return (
        model_choice, rotation_target, full_rate, rotation_profile,
        central_diff_rate, rotation_scale, mapping_precision, 
        newton_precision, spline_order, lagrange_order, max_degree, 
        angular_resolution, plot_resolution, save_name
    )

def give_me_a_name(model_choice, rotation_target) : 
    """
    Constructs a name for the save file using the model name
    and the rotation target.

    Parameters
    ----------
    model_choice : string or Dotdict instance.
        File name or polytrope caracteristics.
    rotation_target : float
        Final rotation rate on the equator.

    Returns
    -------
    save_name : string
        Output file name.

    """
    radical = (
        'poly_' + str(int(model_choice.index)) 
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
        model = polytrope(*MOD_1D.values())
        
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
        r1D, rho1D, *other_var = np.genfromtxt(
            './Models/'+MOD_1D, skip_header=2, unpack=True
        )
        _, idx = np.unique(r1D, return_index=True) 
        N = len(idx)
        
        # Normalisation
        R = r1D[-1]
        M = 4*np.pi * integrate(x=r1D[idx], y=r1D[idx]**2 * rho1D[idx])
        r   = r1D[idx] / (R)
        rho = rho1D[idx] / (M/R**3)
        P0  = surface_pressure / (M**2/R**4)
        
        # We assume that P0 is already normalised by G if R ~ 1 ...
        if not np.allclose(R, 1) :
            P0 /= G
    
    return P0, N, M, R, r, rho, other_var

        
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
    # Mass integration in each direction
    mass_ang = np.array(
        [integrate(rk, rho_n * rk**2, k=5) for rk in map_n.T]   # <- k is set to 5 for 
    )                                                           #    maximal precision
    
    # Integration over the angular domain
    mass_tot = 2*np.pi * mass_ang @ weights    
    
    #### TESTS #####
    
    # import probnum as pn
    # lengthscales = np.logspace(-4, 0, 10)
    # prec = []
    # for l in lengthscales :
    #     mass_ang = [
    #         pn.quad.bayesquad_from_data(
    #             nodes=rk[:, None], 
    #             fun_evals=rho_n * rk**2, 
    #             kernel=pn.randprocs.kernels.ExpQuad(
    #                 input_shape=1, 
    #                 lengthscales=l
    #             ),
    #             domain=(0, rk[-1])
    #         )[0] for rk in map_n.T
    #     ]
    #     extract_moments = lambda rv : (rv.mean, rv.var)
    #     means, vars = np.array(list(map(extract_moments, mass_ang))).T
    #     mass_ang_rv = pn.randvars.Normal(means, np.diag(vars))
    #     mass_rv = 2*np.pi * mass_ang_rv @ weights
    #     prec.append(1.0 - mass_rv.mean)
    # plt.scatter(lengthscales, prec, s=5)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()
    
    # mass_ang = np.array(
    #     [quad(interpolate_func(rk, rho_n * rk**2, der=0), [0.0, rk[-1]]) for rk in map_n.T]
    # )
    # for k in range(1, 6) :
    #     prec = []
    #     orders = np.arange(1,101)
    #     for o in orders : 
    #         mass_ang = np.array(
    #             [GaussQuadrature(rk, rho_n * rk**2, order=o, k=k) for rk in map_n.T]
    #         )
    #         # Integration of the angular domain
    #         mass_tot = 2*np.pi * mass_ang @ weights    
    #         prec.append(abs(1.0 - mass_tot))
    #     plt.scatter(orders, prec, s=5)
    #     mass_ang = np.array(
    #            [integrate(rk, rho_n * rk**2, k=k) for rk in map_n.T]
    #     )
    #     mass_tot = 2*np.pi * mass_ang @ weights  
    #     plt.plot(orders, abs(1.0 - mass_tot)*np.ones_like(orders), lw=1.0) 
    # plt.yscale('log')
    # plt.show()
    
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
    Mlt = lambda l, r : integrate(r, rho_n * r**(l+2), k=5)
    moments_ang = - np.array(
        [[Mlt(l, rk) for rk in map_n.T] for l in degrees]
    )
    pl = np.array([eval_legendre(l, cth) for l in degrees])                                                      
    
    # Integration over the angular domain
    moments = 2*np.pi * (moments_ang * pl) @ weights  
    
    return moments 

def find_pressure(rho, dphi_eff) :
    """
    Find the pressure evaluated on r thanks to the hydrostatic
    equilibrium.

    Parameters
    ----------
    rho : array_like, shape (N, )
        Density profile.
    dphi_eff : array_like, shape (N, )
        Effective potential derivative with respect to r.

    Returns
    -------
    P : array_like, shape (N, )
        Pressure profile.

    """
    dP = - rho * dphi_eff
    P  = interpolate_func(r, dP, der=-1, k=KSPL, prim_cond=(-1, P0))(r)
    return P


def find_rho_l(map_n, rho_n) : 
    """
    Find the density distribution harmonics from a given mapping 
    (map_n) which gives the lines of constant density (=rho_n).

    Parameters
    ----------
    map_n : array_like, shape (N, M)
        Current mapping.
    rho_n : array_like, shape (N, )
        Current density on each equipotential.

    Returns
    -------
    rho_l : array_like, shape (N, L)
        Density distribution harmonics.

    """
    # Density interpolation on the mapping
    all_k   = np.arange((M+1)//2)
    log_rho = np.log(rho_n)
    rho2D   = np.zeros((N, M))
    for k in all_k :
        inside = r < map_n[-1, k]
        rho2D[inside, k] = interpolate_func(
            x=map_n[:, k], y=log_rho, k=KSPL
        )(r[inside])
        rho2D[inside, 0+k] = np.exp(rho2D[inside, k])
    rho2D[:,-1-all_k] = rho2D[:, all_k]
    
    # Corresponding harmonic decomposition
    rho_l = pl_project_2D(rho2D, L)
    
    return rho_l

def filling_ab(ab, ku, kl, l) : 
    """
    Fill the band storage matrix according to the Poisson's
    equation (written in terms of r^2). We optimize this
    operation using the scipy sparse matrix storage (the 
    [::-1] comes from the opposite storage convention 
    between Fortran and Scipy) and exploiting the fact that 
    most of the matrix stay the same when changing l.

    Parameters
    ----------
    ab : array_like, shape (ldmat, 2*N)
        Band storage matrix.
    ku : integer
        Number of terms in the upper matrix part.
    kl : integer
        Number of terms in the lower matrix part.
    l : integer
        Harmonic degree.

    Returns
    -------
    ab : array_like, shape (ldmat, 2*N)
        Filled band storage matrix.

    """    
    # Offset definition
    offset = ku + kl
    
    # The common filling part
    if l == 0 :
        ab[ku+1+(0-0):-1+(0-0):2, 0::2] =  Asp.data[::-1]
        ab[ku+1+(1-0):        :2, 0::2] = -Lsp.data[::-1]
        ab[ku+1+(1-1):-1+(1-1):2, 1::2] =  Dsp.data[::-1]
                
        # First boundary condition (l = 0)
        ab[offset, 0] = 6.0
        
    # The only l dependent part (~1/4 of the matrix)
    else : 
        ab[ku+1+(0-1):-1+(0-1):2, 1::2] = -l*(l+1) * Lsp.data[::-1]
        
        # First boundary condition (l != 0)
        ab[offset-0, 0] = 0.0
        ab[offset-1, 1] = 1.0
    
    # Boundary conditions
    ab[offset+1, 2*N-2] = 2*r[-1]**2
    ab[offset+0, 2*N-1] = l+1 
    return ab
    
    
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
    phi_g_l : array_like, shape (N, L)
        Gravitation potential harmonics.
    phi_eff : array_like, shape (N, )
        Effective potential on each equipotential.
    dphi_eff : array_like, shape (N, ), optional
        Effective potential derivative with respect to r^2.

    """    
    # Density distribution harmonics
    rho_l    = find_rho_l(map_n, rho_n)
    phi_g_l  = np.zeros((N, L))
    dphi_g_l = np.zeros((N, L))
    
    # Vector filling (vectorial)
    Nl = (L+1)//2
    bl = np.zeros((2*N, Nl))
    bl[1:-1:2, :] = 4*np.pi * Lsp @ (r[:,None]**2 * rho_l[:, ::2])
    bl[0     , 0] = 4*np.pi * rho_l[0, 0]     # Boundary condition
    
    # Band matrix storage
    kl = 2*KLAG
    ku = 2*KLAG
    ab = np.zeros((2*kl + ku + 1, 2*N))    
    
    for k in range(Nl) :
        l = 2*k
        
        # Matrix filling  
        ab = filling_ab(ab, ku, kl, l)
        
        # Matrix inversion (LAPACK)
        lub, piv, x, info = dgbsv(kl, ku, ab, bl[:, k])
        if info != 0 : 
            raise ValueError(
                "Problem with finding the gravitational potential. \n",
                "Info = ", info
                )
            
        # Poisson's equation solution
        phi_g_l[: , l] = x[1::2]
        dphi_g_l[:, l] = x[0::2] * (2*r)  # <- The equation is solved on r^2
    
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
    cth : float or array_like, shape (Nr, )
        Value(s) of cos(theta).
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
    else :
        return phi_c, dphi_c


def estimate_omega(phi_g, phi_g_l_surf, target, omega) :
    """
    Estimates the adequate rotation rate so that it reaches ROT
    after normalisation. Considerably speed-up (and stabilises)
    the overall convergence.

    Parameters
    ----------
    phi_g : function(r_eval)
        Gravitational potential along the equatorial cut.
    phi_g_l_surf : array_like, shape (L, )
        Gravitation potential harmonics on the surface.
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
    l    = np.arange(L)
    dr    = 1.0
    r_est = 1.0
    while abs(dr) > DELTA : 
        # Star's exterior
        if r_est >= r[-1] :
            phi_g_l_ext  = phi_g_l_surf * (1.0/r_est)**(l+1)
            dphi_g_l_ext = -(l+1) * phi_g_l_ext / r_est
            phi_g_est  = pl_eval_2D( phi_g_l_ext, 0.0)
            dphi_g_est = pl_eval_2D(dphi_g_l_ext, 0.0)
            
        # Star's interior
        else :
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
        dr = (target - phi_t_est) / dphi_t_est
        r_est += dr
        
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
    # 2D gravitational potential
    phi2D_g =  pl_eval_2D( phi_g_l, cth)
    dphi2D_g = pl_eval_2D(dphi_g_l, cth)
        
    # Gravitational potential interpolation
    l, up = np.arange(L), np.arange((M+1)//2)
    phi_g_func = [CubicHermiteSpline(
        x=r, y=phi2D_g[:, k], dydx=dphi2D_g[:, k]
    ) for k in up]
    
    # Find a new value for ROT
    target = phi_eff[N-1] - pl_eval_2D(phi_g_l[0], 0.0) + phi_eff[0]
    omega_n_new = estimate_omega(phi_g_func[-1], phi_g_l[-1], target, omega_n)
    
    # Find the mapping
    map_est = np.copy(map_n[:, up])
    idx = np.indices(map_est.shape)
    dr = np.ones_like(map_n[:, up])
    dr[0] = 0.0        
    
    while np.any(np.abs(dr) > DELTA) :
        
        # Star's interior
        C_int = (np.abs(dr) > DELTA) & (map_est <  1.0)
        r_int = map_est[C_int]
        k_int = idx[1,  C_int]
        
        if 0 not in k_int.shape :
            
            # Gravitational potential
            inv_sort = np.argsort(app_list(np.arange(len(k_int)), k_int))
            phi_g_int  = np.array(
                (app_list(r_int, k_int, phi_g_func        ),
                 app_list(r_int, k_int, phi_g_func, (1,)*M))
            )[:, inv_sort]
            
            # Centrifugal potential
            phi_c_int = np.array(
                find_centrifugal_potential(r_int, cth[k_int], omega_n_new)
            )
            
            # Total potential
            phi_t_int =  phi_g_int +  phi_c_int
            
            # Update map_est
            dr[C_int] = (phi_eff[idx[0, C_int]] - phi_t_int[0]) / phi_t_int[1]
            map_est[C_int] += dr[C_int]
            
        # Star's exterior
        C_ext = (np.abs(dr) > DELTA) & (map_est >=  1.0)
        r_ext = map_est[C_ext]
        k_ext = idx[1,  C_ext]
        
        if 0 not in k_ext.shape :
            
            # Gravitational potential
            phi_g_l_ext  = phi_g_l[-1] * (r_ext[:, None])**-(l+1)
            dphi_g_l_ext = -(l+1) * phi_g_l_ext / r_ext[:, None]
            inv_sort = np.argsort(app_list(np.arange(len(k_ext)), k_ext))
            phi_g_ext  = np.vstack(
                [app_list(harms, k_ext, pl_eval_2D, cth)
                 for harms in (phi_g_l_ext, dphi_g_l_ext)]
                )[:, inv_sort]
            
            # Centrifugal potential
            phi_c_ext = np.array(
                find_centrifugal_potential(r_ext, cth[k_ext], omega_n_new)
            )
            
            # Total potential
            phi_t_ext =  phi_g_ext +  phi_c_ext
            
            # Update map_est
            dr[C_ext] = (phi_eff[idx[0, C_ext]] - phi_t_ext[0]) / phi_t_ext[1]
            map_est[C_ext] += dr[C_ext]
            
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
    


#%% Main cell

if __name__ == '__main__' :
    
    start = time.perf_counter()
    
    # Definition of global parameters
    MOD_1D, ROT, FULL, PROFILE, ALPHA, SCALE, EPS, DELTA, KSPL, \
        KLAG, L, M, RES, SAVE = set_params() 
    G = 6.67384e-8     # <- value of the gravitational constant
        
    # Definition of the 1D-model
    P0, N, mass, radius, r, rho_n, VAR = init_1D()  
    
    # Angular domain preparation
    cth, weights, map_n      = init_2D()
    
    # Centrifugal potential definition
    eval_phi_c = init_phi_c()
    
    # Find the lagrange matrix
    lag_mat = lagrange_matrix_P(r**2, order=KLAG)
    
    # Define sparse matrices 
    Lsp = sps.dia_matrix(lag_mat[..., 0])
    Dsp = sps.dia_matrix(lag_mat[..., 1])
    Asp = sps.dia_matrix(
        4 * lag_mat[..., 1] * r**4 - 2 * lag_mat[..., 0] * r**2
    )
    
    # Initialisation for the effective potential
    phi_g_l, dphi_g_l, phi_eff, dphi_eff = find_phi_eff(map_n, rho_n)
    
    # Find pressure
    P = find_pressure(rho_n, dphi_eff)
    
    # Iterative centrifugal deformation
    surfaces = [map_n[-1]]
    r_pol = [0.0, find_r_pol(map_n)]
    n = 1
    
    # SAVE
    phi_g_l_rad  = [phi_g_l]
    dphi_g_l_rad = [dphi_g_l]
    phi_eff_rad  = [np.copy(phi_eff)]
    map_n_rad    = [map_n]
    omega_n_rad  = [0.0]
    
    
    while abs(r_pol[-1] - r_pol[-2]) > EPS :
        
        # Current rotation rate
        omega_n = min(ROT, (n/FULL) * ROT)
        
        # Effective potential computation
        phi_g_l, dphi_g_l, phi_eff = find_phi_eff(map_n, rho_n, phi_eff)
        
        # SAVE
        phi_g_l_rad.append(phi_g_l)
        dphi_g_l_rad.append(dphi_g_l)
        phi_eff_rad.append(np.copy(phi_eff))

        # Update the mapping
        map_n, omega_n = find_new_mapping(map_n, omega_n, phi_g_l, dphi_g_l, phi_eff)

        # Renormalisation
        r_corr    = find_r_eq(map_n)
        m_corr    = find_mass(map_n, rho_n)
        radius   *= r_corr
        mass     *= m_corr
        map_n    /=             r_corr
        rho_n    /= m_corr    / r_corr**3
        phi_eff  /= m_corr    / r_corr
        dphi_eff /= m_corr    / r_corr**2
        P        /= m_corr**2 / r_corr**4
        
        # Update the surface and polar radius
        surfaces.append(map_n[-1])
        r_pol.append(find_r_pol(map_n))
        
        # SAVE
        map_n_rad.append(map_n)
        omega_n_rad.append(omega_n)
        
        # Iteration count
        DEC = int(-np.log10(EPS))
        print(f"Iteration n°{n}, R_pol = {r_pol[-1].round(DEC)}")
        n += 1
    
    finish = time.perf_counter()
    print(f'Deformation done in {round(finish-start, 4)} sec')   
    
    # Plot mapping
    plot_f_map(map_n, rho_n, phi_eff, L, show_surfaces=True)        
        
    # Compute the rotation profile 
    # eval_w = la_bidouille('rota_eq.txt', smoothing=1e-5, return_profile=True)
    # w_eq, dw_eq = eval_w(map_n[:, (M-1)//2], 0.0, ROT)
    
    # Model scaling
    map_n    *=               radius
    rho_n    *=     mass    / radius**3
    phi_eff  *= G * mass    / radius   
    dphi_eff *= G * mass    / radius**2
    P        *= G * mass**2 / radius**4
    
    # Model writing
    # write_model(SAVE, map_n, r, P, rho_n, phi_eff, w_eq, dw_eq)
    
        