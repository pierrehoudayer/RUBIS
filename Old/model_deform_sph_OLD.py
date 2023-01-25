#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 18:36:42 2022

@author: phoudayer
"""

#%% Modules cell

import orthopy
import quadpy
import matplotlib.colors as mcl
import matplotlib.pyplot as plt
import numpy             as np
import scipy.sparse      as sps
from matplotlib             import rc
from matplotlib.collections import LineCollection
from pylab                  import cm
from scipy.interpolate      import splrep, splantider, splev, splint
from scipy.linalg.lapack    import dgbsv
from scipy.special          import expn

#%% Classes

class DotDict(dict):  
    """dot.notation access to dictionary attributes"""      
    def __getattr__(*args):        
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val     
    __setattr__ = dict.__setitem__     
    __delattr__ = dict.__delitem__ 
    

#%% Low-level functions cell

def expinv(x, k=1, a=1) : 
    """
    Function returning the value of:
        \mathrm{Expinv}(x, k) = \exp\left(-x^{-1/k}\right)
    
    This function is an analytical continuation for x >= 0 of the
    function f(x) = 0 for x < 0, giving it a "plateau" looking shape
    close to 0. The length of this plateau can be changed by modifying
    the value of k (higher k, smaller plateau)

    Parameters
    ----------
    x : FLOAT or ARRAY
        Input value
    k : FLOAT, optional
        Exponent value, it can theoritically be a real but must
        be keep to an integer if one want to compute the analytical
        primitive of the function (multivalued otherwise). 
        The default is 1.
    a : FLOAT, optional
        Add an optional homotesy to the axis. The default is 1.

    Returns
    -------
    y : FLOAT or ARRAY (same shape as x)
        Output value

    """
    with np.errstate(all='ignore') :
        u = x**-(1/k)
        y = np.exp(-a*u)
        return y

def expI(x, k=1, a=1) : 
    """
    Useful function for computing the primitive of expinv(x, k). It 
    is defined as: 
        \mathrm{Exp}_\mathrm{I}(x, k) = kE_{k+1}\left(x^{-1/k}\right)
        
    with E_n(x) the generalised exponential integral:
        E_{n}\left(x\right) = 
        x^{n-1}\int_{x}^{\infty}\frac{e^{-t}}{t^{n}}\mathrm{d}x.        

    Parameters
    ----------
    x : FLOAT or ARRAY
        Input value
    k : INT, optional
        Argument of expinv(x, k). The default is 1.
    a : FLOAT, optional
        Add an optional homotesy to the axis. The default is 1.

    Returns
    -------
    y : FLOAT or ARRAY (same shape as x)
        Output value

    """
    with np.errstate(all='ignore') :
        u = x**-(1/k)
        y = k * expn(k+1, a*u)
        return y
    

def integrate(x, y, a=None, b=None, k=3) : 
    """
    Function computing the integral of f(x) between a and b
    for fixed sampled values of y_i = f(x_i) at x_i. The routine 
    makes use of the scipy.interpolate.splXXX functions to perform
    this integral using B-splines.

    Parameters
    ----------
    x : ARRAY (N, )
        x values on which to integrate.
    y : ARRAY (N, )
        y values to integrate.
    a, b : FLOATS, optional
        Lower and upper bounds used to compute the integral. 
        The default is None.
    k : INT, optional
        Degree of the B-splines used to compute the integral. 
        The default is 3.

    Returns
    -------
    integral : FLOAT
        Result of the integration.

    """
    tck = splrep(x, y, k=k)
    if a == None :
        a = x[0]
    if b == None : 
        b = x[-1]
    integral = splint(a, b, tck)
    return integral

def interpolate_func(x, y, der=0, k=3, s=0, prim_cond=None):
    """
    Routine returning an interpolation function of (x, y) 
    for a given B-spline order k. A derivative order can 
    be specified by the value der < k 
    (if der=-1, returns an antiderivative).

    Parameters
    ----------
    x : ARRAY (N, )
        x values on which to integrate.
    y : ARRAY (N, )
        y values to integrate.
    der : INT, optional
        Order of the derivative. 
        The default is 0.
    k : INT, optional
        Degree of the B-splines used to compute the integral. 
        The default is 3.
    s : FLOAT, optional
        Smoothing parameter. 
        The default is 0.
    prim_cond : ARRAY (2, ), optional
        Conditions to specify the constant to add to the
        primitive function if der = -1. The first value 
        is an integer i, such that F(x[i]) = second value.
        The default is None, which correspond to F(x[0]) = 0.

    Returns
    -------
    func : FUNC(x_eval)
        Interpolation function of x_eval.

    """
    tck = splrep(x, y, k=k, s=s)
    if not type(der) is int :
        raise TypeError(
            f"""Only integers are allowed for the derivative order. 
            Current value is {der}."""
            )
    if not -2 < der < k : 
        raise ValueError(
            f"""Derivative order should be either -1 (antiderivative)
            or 0 <= der < k={k} (derivative). Current value is {der}."""
            )
    if der >= 0 :
        def func(x_eval) :
            if np.array(x_eval).shape != (0,) :
                return splev(x_eval, tck, der=der)
            else : 
                return np.array([])
        return func
    else :
        tck_antider = splantider(tck)
        cnst = 0.0
        if prim_cond is not None :
            cnst = prim_cond[1] - splev(x[prim_cond[0]], tck_antider)
        def func(x_eval) :
            return splev(x_eval, tck_antider) + cnst
        return func
    
def scheme_GaussLegendre(n):
    """
    Function returning the scheme (containing the ticks and weights)
    required to the Gauss-Legendre quadrature.

    Parameters
    ----------
    n : INT
        Quadrature order.

    Returns
    -------
    scheme : OBJ
        Object containing the information regarding the 
        integration scheme. The points can be obtained
        with scheme.points and the weights with
        scheme.weights.

    """
    return quadpy.c1.gauss_legendre(n)

def lagrange_matrix(x, order=2) :
    """
    Computes the interpolation and derivation matrices based on
    an initial grid x. The new grid, result.grid, on which the 
    interpolation/derivation takes place is entierly defined 
    from the x nodes following Reese (2013).

    Parameters
    ----------
    x : ARRAY (N, )
        Initial grid from which one interpolates/derives
    order : INT, optional
        Scheme order from the lagrange interpolation/derivation.
        The effective precision order is 2*order, even though the 
        number of points involved in each window is also 2*order.
        The default is 2.

    Returns
    -------
    result : DotDict(keys=[mat, grid])
        mat : ARRAY(N-1, N, 2)
            Contains the interpolation (mat[...,0]) and
            the derivation (mat[...,1]) matrices.
        grid : ARRAY (N-1, )
            Grid on which the interpolation/derivation takes
            place.

    """
    from init_derive_IFD import init_derive_ifd
    mat_lag, x_lag = init_derive_ifd(x, 1, order)
    result = DotDict(
        dict(mat = mat_lag, grid = x_lag)
        )
    return result        


#%% Centrifugal potential cell

def solid(r, cth, omega) :
    """
    Computes the centrifugal potential and its derivative in the 
    case of a solid rotation profile.

    Parameters
    ----------
    r : FLOAT or ARRAY
        Distance from the origin.
    cth : FLOAT
        Value of cos(theta).
    omega : FLOAT
        Rotation rate.

    Returns
    -------
    phi_c : FLOAT or ARRAY (same shape as r)
        Centrifugal potential.
    dphi_c : FLOAT or ARRAY (same shape as r)
        Centrifugal potential derivative with respect to r.

    """
    s2 = 1 - cth**2
    phi_c  = -0.5 * r**2 * s2 * omega**2
    dphi_c = -1.0 * r**1 * s2 * omega**2 
    return phi_c, dphi_c

def lorentzian_profile(r, cth, omega) :
    """
    Lorentzian rotation profile. The rotation rate difference 
    between the center (c) and the equator (eq) is fixed by 
    ALPHA = \frac{\Omega_c - \Omega_\mathrm{eq}}{\Omega_\mathrm{eq}}.

    Parameters
    ----------
    r : FLOAT or ARRAY
        Distance from the origin.
    cth : FLOAT
        Value of cos(theta).
    omega : FLOAT
        Rotation rate on the equator.

    Returns
    -------
    ws : FLOAT or ARRAY (same shape as r)
        Rotation rate at (r, cth).

    """
    s2 = r**2 * (1 - cth**2)   
    ws = (1 + ALPHA) / (1 + ALPHA*s2) * omega
    return ws

def lorentzian(r, cth, omega) :
    """
    Computes the centrifugal potential and its derivative in the 
    case of a lorentzian rotation profile (cf. function
    lorentzian_profile(r, cth, omega)).

    Parameters
    ----------
    r : FLOAT or ARRAY
        Distance from the origin.
    cth : FLOAT
        Value of cos(theta).
    omega : FLOAT
        Rotation rate on the equator.

    Returns
    -------
    phi_c : FLOAT or ARRAY (same shape as r)
        Centrifugal potential.
    dphi_c : FLOAT or ARRAY (same shape as r)
        Centrifugal potential derivative with respect to r.

    """
    s2, ds2 = r**2 * (1 - cth**2), 2*r * (1 - cth**2)
    phi_c  = -0.5 * s2  * (1 + ALPHA)**2 / (1 + ALPHA*s2)**1 * omega**2
    dphi_c = -0.5 * ds2 * (1 + ALPHA)**2 / (1 + ALPHA*s2)**2 * omega**2
    return phi_c, dphi_c

def plateau_profile(r, cth, omega, k=1) : 
    """
    Rotation profile with a "plateau" close to s = 0. It can be used
    to simulate a solid rotation in the center. The rotation rate difference 
    between the center (c) and the equator (eq) is fixed by 
    ALPHA = \frac{\Omega_c - \Omega_\mathrm{eq}}{\Omega_\mathrm{eq}}. The 
    profile scale can be adjusted using the global parameter SCALE.

    Parameters
    ----------
    r : FLOAT or ARRAY
        Distance from the origin.
    cth : FLOAT
        Value of cos(theta).
    omega : FLOAT
        Rotation rate on the equator.
    k : INT, optional
        Value that impacts the plateau length (higher k, smaller plateau). 
        The default is 1.

    Returns
    -------
    ws : FLOAT or ARRAY (same shape as r)
        Rotation rate at (r, cth).

    """
    corr = np.exp(SCALE**(2/k))
    w0 = (1 + ALPHA) * omega
    dw = ALPHA * omega * corr
    x  = r**2 * (1 - cth**2) / SCALE**2
    ws = w0 - dw * expinv(x, k)
    return ws

def plateau(r, cth, omega, k=1) :
    """
    Computes the centrifugal potential and its derivative in the 
    case of a "plateau" rotation profile (cf. function
    plateau_profile(r, cth, omega)).

    Parameters
    ----------
    r : FLOAT or ARRAY
        Distance from the origin.
    cth : FLOAT
        Value of cos(theta).
    omega : FLOAT
        Rotation rate on the equator.
    k : INT, optional
        Value that impacts the plateau length (higher k, smaller plateau). 
        The default is 1.

    Returns
    -------
    phi_c : FLOAT or ARRAY (same shape as r)
        Centrifugal potential.
    dphi_c : FLOAT or ARRAY (same shape as r)
        Centrifugal potential derivative with respect to r.

    """
    corr = np.exp(SCALE**(2/k))
    w0 = (1 + ALPHA) * omega
    dw = ALPHA * omega * corr
    s2, ds2 = r**2 * (1 - cth**2), 2*r * (1 - cth**2)
    x = s2 / SCALE**2
    I1 , I2  = expI(  x, k, 1), expI(  x, k, 2)
    II1, II2 = expinv(x, k, 1), expinv(x, k, 2)
    phi_c  = -0.5 * s2  * (w0**2 - 2*w0*dw * I1  + dw**2 * I2 )
    dphi_c = -0.5 * ds2 * (w0**2 - 2*w0*dw * II1 + dw**2 * II2)
    return phi_c, dphi_c

def la_bidouille(fname, smoothing=0) : 
    """
    Sets up the function phi_c_func(r, cth, omega) which computes
    the centrifugal potential and its derivative using a numerical 
    rotation profile strored in fname. The interpolation, integration,
    smoothing is handled by a spline decomposition routine.
    
    Parameters
    ----------
    fname : STR
        File to be read.
    smoothing : FLOAT, optional
        Optional smoothing value applied to the numerical rotation
        profile. 
        The default is 0.

    Returns
    -------
    phi_c_func : FUNC(r, cth, omega)
        cf. below

    """
    dat = np.loadtxt(fname)
    _, idx = np.unique(dat[:, 0], return_index=True)
    sd, wd, _ = dat[idx].T
    phi_c_int  = interpolate_func(sd, -sd * wd**2, der=-1, s=smoothing)
    dphi_c_int = interpolate_func(sd, -sd * wd**2, der= 0, s=smoothing)
    
    def phi_c_func(r, cth, omega) : 
        """
        Computes the centrifugal potential and its derivative 
        using a numerical rotation profile.

        Parameters
        ----------
        r : FLOAT or ARRAY
            Distance from the origin.
        cth : FLOAT
            Value of cos(theta).
        omega : FLOAT
            Rotation rate on the equator.

        Returns
        -------
        phi_c : FLOAT or ARRAY (same shape as r)
            Centrifugal potential.
        dphi_c : FLOAT or ARRAY (same shape as r)
            Centrifugal potential derivative with respect to r.

        """
        s = (1 - cth**2)**0.5
        corr = (omega/0.8)**2
        phi_c  =  phi_c_int(r * s) * corr
        dphi_c = dphi_c_int(r * s) * corr * s
        return phi_c, dphi_c
    
    return phi_c_func

#%% High-level functions cell
    
def set_params() : 
    """
    Function returning the main parameters of the file.

    Returns
    -------
    filename : STR 
        Name of the file containing the 1D model.
    rotation_target : FLOAT
        Target for the rotation rate.
    full_rate : INT
        Number of iterations before reaching full rotation rate. As a 
        rule of thumb, ~ 1 + int(-np.log(1-rotation_target)) iterations
        should be enough to ensure convergence (in the solid rotation case!).
    rotation_profile : FUNC(r, cth, omega)
        Function used to compute the centrifugal potential and its 
        derivative. Possible choices are {solid, lorentzian, plateau}.
        Explanations regarding this profiles are available in the 
        corresponding functions.
    rate_difference : FLOAT
        The rotation rate difference between the centre and equator in the
        cylindrical rotation profile. For instance, rate_difference = 0.0
        would corresond to a solid rotation profile while 
        rate_difference = 0.5 indicates that the star's centre rotates 50%
        faster than the equator.
        Only appears in cylindrical rotation profiles.
    rotation_scale : FLOAT
        Homotesy factor on the x = r*sth / Req axis for the rotation profile.
        Only appear in the plateau rotation profile.
    precision_target : FLOAT
        Precision target for the convergence criterion.
    spline_order : INT
        Choice of B-spline order in integration / interpolation
        routines. 
        3 is recommanded (must be odd in anycase).
    lagrange_order : INT
        Choice of Lagrange polynomial order in integration / interpolation
        routines. 
        2 should be enough.
    max_degree : INT
        Maximum l degree to be considered in order to do the
        harmonic projection.
    angular_resolution : INT
        Angular resolution for the mapping. Better to take an odd number 
        in order to include the equatorial radius.
    save_resolution : INT
        Angular resolution for saving the mapping.
    """
    
    #### RAD PARAMETERS ####
    # filename = "unknown_model.txt"              
    # filename = "polytrope1.txt"  
    filename = "polytrope3.txt"            
    # filename = "1Dmodel_1.977127.txt"            
    rotation_target = 0.3
    full_rate = 1 + int(-np.log(1-rotation_target))
    # full_rate = 10
    rotation_profile = solid
    central_diff_rate = 0.2667
    rotation_scale = 0.1
    precision_target = 1e-10
    spline_order = 3
    lagrange_order = 2
    max_degree = angular_resolution = 101
    save_resolution = 501
    
    return (
        filename, rotation_target, full_rate, rotation_profile,
        central_diff_rate, rotation_scale, precision_target, spline_order, 
        lagrange_order, max_degree, angular_resolution, save_resolution
        )

def read_1D() : 
    """
    Function reading the 1D model file 'NAME_1D'

    Returns
    -------
    surface_pressure : FLOAT
        Value of the surface pressure.
    N : INT
        Radial resolution of the model.
    r : ARRAY (N, )
        Radial coordinate.
    rho : ARRAY (N, )
        Radial density of the model.

    """
    surface_pressure, N = np.genfromtxt(
        NAME_1D, max_rows=2, unpack=True
        )
    N = int(N)
    r, rho = np.genfromtxt(
        NAME_1D, skip_header=2, unpack=True
        )
    return surface_pressure, N, r, rho

def renormalise_1D(r1D, rho1D) :
    """
    Function renormalising the radial coordinate and density profile, 
    as well as returning the overall mass and radius of the model.

    Parameters
    ----------
    r1D : ARRAY (N, )
        Radial coordinate.
    rho1D : ARRAY (N, )
        Radial density of the model.

    Returns
    -------
    M : FLOAT
        Total mass of the model.
    R : FLOAT
        Radius of the model.
    r : ARRAY (N, )
        Radial coordinate after normalisation.
    zeta : ARRAY(N, )
        Spheroidal coordinate.
    rho : ARRAY (N, )
        Radial density of the model after normalisation.

    """
    R = r1D[-1]
    r = r1D/R
    zeta = r
    M = 4*np.pi*R**3 * integrate(x=r, y=r**2 * rho1D)
    rho = rho1D * (R**3 / M)
    return M, R, r, zeta, rho
        
def init_2D() :
    """
    Init function for the angular domain.

    Parameters
    ----------
    None.

    Returns
    -------
    cth : ARRAY (M, )
        Angular coordinate (equivalent to cos(theta)).
    weights : ARRAY (M, )
        Angular weights for the Legendre quadrature.
    map_n : ARRAY (N, M)
        Isopotential mapping 
        (given by r(phi_eff, theta) = r for now).
    """
    scheme = scheme_GaussLegendre(M)
    cth, weights = scheme.points, scheme.weights
    map_n = np.tile(r, (M, 1)).T
    return cth, weights, map_n   

def find_r_eq(map_n, cth, weights) :
    """
    Function to find the equatorial radius from the mapping.

    Parameters
    ----------
    map_n : ARRAY (N, M)
        Isopotential mapping.

    Returns
    -------
    r_eq : FLOAT
        Equatorial radius.

    """
    surf_l = pl_project_2D(map_n[-1], cth, weights)
    return pl_eval_2D(surf_l, 0.0)

def find_r_pol(map_n, cth, weights) :
    """
    Function to find the polar radius from the mapping.

    Parameters
    ----------
    map_n : ARRAY (N, M)
        Isopotential mapping.

    Returns
    -------
    r_eq : FLOAT
        Equatorial radius.

    """
    surf_l = pl_project_2D(map_n[-1], cth, weights)
    return pl_eval_2D(surf_l, 1.0)
    
def find_mass(map_n, rho_n, weights) :
    """
    Find the total mass over the mapping map_n.

    Parameters
    ----------
    map_n : ARRAY (N, M)
        Isopotential mapping.
    rho_n : ARRAY (N, )
        Density profile (the same in each direction).
    weights : ARRAY (M, )
        Angular weights for the Legendre quadrature.

    Returns
    -------
    mass_tot : FLOAT
        Total mass integrated of map_n.

    """
    # Find the metrics terms 
    dr = find_metric_terms(map_n)
    
    # Starting by the computation of the mass in each angular direction
    mass_ang = np.array(
        [integrate(
            zeta, rho_n * dr._[:, k]**2 * dr.z[:, k] , k=SPL_ORDER
            ) for k in range(M)]
        )
    # Integration of the angular domain
    mass_tot = 2*np.pi * sum(mass_ang * weights)
    return mass_tot

def find_pressure(rho, dphi_eff) :
    """
    Find the pressure evaluated on r thanks to the hydrostatic
    equilibrium.

    Parameters
    ----------
    rho : ARRAY (N, )
        Density profile.
    dphi_eff : ARRAY (N, )
        Effective potential derivative with respect to r^2.

    Returns
    -------
    P : ARRAY (N, )
        Pressure profile.

    """
    dP = - (2*r) * rho * dphi_eff
    P  = interpolate_func(
        r, dP, der=-1, k=SPL_ORDER, prim_cond=(-1, P0)
        )(r)
    return P

def pl_project_2D(f, cth, weights) :
    """
    Projection of function, assumed to be already evaluated 
    at the Gauss-Legendre scheme points, over the Legendre 
    polynomials.    

    Parameters
    ----------
    f : ARRAY (N, M)
        function to project.
    cth : ARRAY (M, )
        Gauss-Legendre scheme points.
    weights : ARRAY (M, )
        Gauss-Legendre scheme weights.

    Returns
    -------
    f_l : ARRAY (N, L)
        The projection of f over the legendre polynomials
        for each radial value.

    """
    pl_series = orthopy.c1.legendre.Eval(cth, "normal")
    f_l = np.array(
        [f @ (weights * next(pl_series)) for l in range(0, L)]
        ).T
    return f_l

def pl_eval_2D(f_l, t, der=0) :
    """
    Evaluation of f(r, t) (and its derivatives) from a projection,
    f_l(r, l), of f over the Legendre polynomials.

    Parameters
    ----------
    f_l : ARRAY(N, L)
        The projection of f over the legendre polynomials.
    t : ARRAY(N_t, )
        The points on which to evaluate f.
    der : INT in {0, 1, 2}
        The upper derivative order. The default value is 0.
    Returns
    -------
    f : ARRAY(N, N_t)
        The evaluation of f over t.
    df : ARRAY(N, N_t), optional
        The evaluation of the derivative f over t.
    ddf : ARRAY(N, N_t), potional
        The evaluation of the 2nd derivative of f over t.

    """
    assert der in {0, 1, 2} # Check the der input
    
    # f computation
    pl_series = orthopy.c1.legendre.Eval(t, "normal")
    all_pl = np.array([next(pl_series) for l in range(L)])
    f = f_l @ all_pl
    
    if der != 0 :
        # df computation
        all_l = np.arange(L)
        norm = np.sqrt(2 / (2*all_l+1))
        all_dpl = all_l[:, None] * np.roll(norm[:, None] * all_pl, 1, axis=0)
        for l in range(1, L) : 
            all_dpl[l] += t * all_dpl[l-1]
        all_dpl /= norm[:, None]
        df = f_l @ all_dpl
        
        if der != 1 :
            # ddf computation
            all_lp1 = np.where(all_l != 0, all_l+1, 0)
            all_ddpl = all_lp1[:, None] * np.roll(
                norm[:, None] * all_dpl, 1, axis=0
                )
            for l in range(1, L) : 
                all_ddpl[l] += t * all_ddpl[l-1]
            all_ddpl /= norm[:, None]
            ddf = f_l @ all_ddpl
            
            return f, df, ddf
        return f, df
    return f

def find_phi_eff(map_n, rho_n, omega_n, cth, weights, phi_eff=None) :
    """
    Determination of the effective potential from a given mapping
    (map_n, which gives the lines of constant density), and a given 
    rotation rate (omega_n). This potential is determined by solving
    the Poisson's equation on each degree of the harmonic decomposition
    (giving the gravitational potential harmonics which are also
    returned) and then adding the centrifugal potential.

    Parameters
    ----------
    map_n : ARRAY(N, M)
        Current mapping.
    rho_n : ARRAY(N, )
        Current density on each equipotential.
    omega_n : FLOAT
        Current rotation rate.
    cth : ARRAY(M, )
        Value of cos(theta) in the Gauss-Legendre scheme.
    weights : ARRAY(M, )
        Integration weights in the Gauss-Legendre scheme.
    phi_eff : ARRAY(N, ), optional
        If given, the current effective potential on each 
        equipotential. If not given, it will be calculated inside
        this fonction. The default is None.

    Raises
    ------
    ValueError
        If the matrix inversion enconters a difficulty ...

    Returns
    -------
    phi_g_l : ARRAY(N, L)
        Gravitation potential harmonics.
    phi_eff : ARRAY(N, )
        Effective potential on each equipotential.
    dphi_eff : ARRAY(N, ), optional
        Effective potential derivative with respect to zeta.

    """    
    # Empty harmonics initialisation
    phi_g_l  = np.zeros((N, L))
    dphi_g_l = np.zeros((N, L))
    
    # Metric terms and coupling integral computation
    dr = find_metric_terms(map_n)
    Pll = find_all_couplings(dr, cth, weights)
    
    # Vector filling (vectorial)
    Nl = (L+1)//2
    b = np.zeros(2*N*Nl)
    r2rz_l = pl_project_2D(dr._**2 * dr.z, cth, weights) 
    b[Nl:-Nl:2] = 4*np.pi * (
        Lsp @ (rho_n[:, None] * r2rz_l[:, ::2])
        ).reshape((-1))
    
    
    # Band matrix storage
    kl = (2*LAG_ORDER + 1) * Nl - 1
    ku = (2*LAG_ORDER + 1) * Nl - 1
    ab = np.zeros((2*kl+ku+1, 2*N*Nl))
    
    # Main part terms
    temp = np.empty((2*LAG_ORDER, N, 2*Nl, 2*Nl))
    temp[..., 0::2, 0::2] = (
        + Dsp.data[::-1, :, None, None] * Pll.zz 
        - Lsp.data[::-1, :, None, None] * Pll.zt
        )
    temp[..., 0::2, 1::2] = (
        - Lsp.data[::-1, :, None, None] * Pll.tt
        )
    temp[..., 1::2, 0::2] = (
        + Lsp.data[::-1, :, None, None] * np.eye(Nl)
        )
    temp[..., 1::2, 1::2] = (
        - Dsp.data[::-1, :, None, None] * np.eye(Nl)
        )
    ab[kl+L:] = np.moveaxis(temp, 2, 1).reshape((2*LAG_ORDER*2*Nl, 2*N*Nl))
    del temp
        
    # Inner boundary conditions 
    ab[kl+ku:kl+ku+Nl, 0:2*Nl:2] = np.diag((1, ) + (0, )*(Nl-1))
    ab[kl+ku:kl+ku+Nl, 1:2*Nl:2] = np.diag((0, ) + (1, )*(Nl-1))
    
    # Outer boundary conditions
    BC0, BC1 = find_outer_BC(dr, cth, weights)  
    ab[kl+ku+Nl:kl+ku+2*Nl, -2*Nl+0::2] = BC0
    ab[kl+ku+Nl:kl+ku+2*Nl, -2*Nl+1::2] = BC1
    
    # Matrix reindexing
    for l in range(1, 2*Nl) : 
        ab[kl+L-l:-l, l::2*Nl] = ab[kl+L:, l::2*Nl]
        ab[-l:, l::2*Nl] = 0.0
                
    # Matrix inversion (LAPACK)
    _, _, x, info = dgbsv(kl, ku, ab, b)

    if info != 0 : 
        raise ValueError(
            "Problem with finding the gravitational potential. \n",
            "Info = ", info
            )
            
    # Poisson's equation solution
    phi_g_l[: , ::2] = x[1::2].reshape((N, Nl))
    dphi_g_l[:, ::2] = x[0::2].reshape((N, Nl))    
    
    if phi_eff is None :
        # First estimate of the effective potential ...
        phi_g   = pl_eval_2D(phi_g_l, 0.0)
        phi_c   = np.zeros_like(phi_g)
        phi_eff = phi_g + phi_c    
        
        # ... and of its derivative
        dphi_g   = pl_eval_2D(dphi_g_l, 0.0)
        dphi_c   = np.zeros_like(dphi_g)
        dphi_eff = dphi_g + dphi_c
        
        return phi_g_l, phi_eff, dphi_eff
        
    # The effective potential is known to an additive constant 
    C = pl_eval_2D(phi_g_l[0], 0.0) - phi_eff[0]
    phi_eff += C
    return phi_g_l, phi_eff
    

def find_centrifugal_potential(r, cth, omega, dim=False) :
    """
    Determination of the centrifugal potential and its 
    derivative in the case of a cylindric rotation profile 
    (caracterised by ALPHA). The option dim = True allows a
    computation taking into account the future renormalisation
    (in this case r_eq != 1 but r_eq = R_eq / R).

    Parameters
    ----------
    r : FLOAT or ARRAY(Nr, )
        Radial value(s).
    cth : FLOAT
        Value of cos(theta).
    omega : FLOAT
        Rotation rate.
    dim : BOOL, optional
        Set to true for the omega computation. 
        The default is False.

    Returns
    -------
    phi_c : FLOAT or ARRAY(Nr, )
        Centrifugal potential.
    dphi_c : FLOAT or ARRAY(Nr, )
        Centrifugal potential derivative with respect to r.

    """
    phi_c, dphi_c = eval_phi_c(r, cth, omega)
    if dim :
        return phi_c / r**3, (r*dphi_c - 3*phi_c) / r**4
    return phi_c, dphi_c


def estimate_omega(map_eq, phi_eff, phi_g, phi_g_l_surf, cth, omega_n) :
    """
    Estimates the adequate rotation rate so that it reaches ROT
    after normalisation. Considerably speed-up (and stabilises)
    the overall convergence. This function has the same signature
    than find_mapping_theta() on purpose (maybe these two routines
    should be merged?).

    Parameters
    ----------
    map_eq : ARRAY(N, )
        Initial guess for the equatorial mapping.
    phi_eff : ARRAY(N, )
        Effective potential on the isopotential. Used to find the
        target value
    phi_g : ARRAY(N, )
        Gravitational potential in the radial direction.
    phi_g_l_surf : ARRAY(L, )
        Gravitation potential harmonics on the surface.
    cth : FLOAT
        Value of cos(theta) in the corresponding angular direction
        (0 in this case).
    omega_n : FLOAT
        Current rotation rate.

    Returns
    -------
    omega_n_new : FLOAT
        New rotation rate.

    """
    # Gravitational potential interpolation
    phi_g_func  = interpolate_func(x=map_eq, y=phi_g, der=0, k=SPL_ORDER)
    dphi_g_func = interpolate_func(x=map_eq, y=phi_g, der=1, k=SPL_ORDER) 
    
    # Searching for a new omega
    l    = np.arange(L)
    target = phi_eff[-1] - phi_g[0] + phi_eff[0]
    dr    = 1.0
    r_est = 1.0
    while abs(dr) > EPS : 
        # Star's exterior
        if r_est >= 1.0 :
            phi_g_l_ext  = phi_g_l_surf * r_est**-(l+1)
            dphi_g_l_ext = -(l+1) * phi_g_l_ext / r_est
            phi_g_est  = pl_eval_2D( phi_g_l_ext, cth)
            dphi_g_est = pl_eval_2D(dphi_g_l_ext, cth)
            
        # Star's interior
        else :
            phi_g_est  =  phi_g_func(r_est)
            dphi_g_est = dphi_g_func(r_est)
        
        # Centrifugal potential 
        phi_c_est, dphi_c_est = find_centrifugal_potential(
            r_est, cth, omega_n, dim=True
            )
        
        # Total potential
        phi_t_est  =  phi_g_est +  phi_c_est
        dphi_t_est = dphi_g_est + dphi_c_est
        
        # Update r_est
        dr = -(phi_t_est - target) / dphi_t_est
        r_est += dr
        
    # Updating omega
    omega_n_new = omega_n * r_est**(-1.5)
    return omega_n_new

    
def find_mapping_theta(guess, phi_eff, phi_g, phi_g_l_surf, cth, omega) :
    """
    Find the mapping in a given angular direction (determined by
    the cth value).

    Parameters
    ----------
    guess : ARRAY(N_phi, )
        Initial guess for the mapping.
    phi_eff : ARRAY(N_phi, )
        Target values for the total potential on r_est.
    phi_g : ARRAY(N, )
        Gravitational potential in the radial direction.
    phi_g_l_surf : ARRAY(L, )
        Gravitation potential harmonics on the surface.
    cth : FLOAT
        Value of cos(theta) in the corresponding angular direction.
    omega_n : FLOAT
        Current rotation rate.

    Returns
    -------
    r_est : ARRAY(N_phi, )
        Resulting mapping.

    """
    # Gravitational potential interpolation
    phi_g_func  = interpolate_func(x=guess, y=phi_g, der=0, k=SPL_ORDER)
    dphi_g_func = interpolate_func(x=guess, y=phi_g, der=1, k=SPL_ORDER) 
    
    # Find the mapping
    l = np.arange(L)
    target = phi_eff
    surf = guess[-1]
    dr = np.ones_like(guess)
    dr[0] = 0.0
    r_est = np.copy(guess)
    while np.any(np.abs(dr) > EPS) :
        
        # Star's interior
        i_int  = (np.abs(dr) > EPS) & (r_est <  surf)
        phi_g_int  =  phi_g_func(r_est[i_int])
        dphi_g_int = dphi_g_func(r_est[i_int])
            
        # Star's exterior
        i_ext = (np.abs(dr) > EPS) & (r_est >= surf)
        phi_g_l_ext  = phi_g_l_surf * (surf / r_est[i_ext, None])**(l+1)
        dphi_g_l_ext = -(l+1) * phi_g_l_ext / r_est[i_ext, None]
        phi_g_ext    = pl_eval_2D(phi_g_l_ext,  cth)
        dphi_g_ext   = pl_eval_2D(dphi_g_l_ext, cth)
        
        # Concatenate
        phi_g_est  = np.concatenate(( phi_g_int,  phi_g_ext))
        dphi_g_est = np.concatenate((dphi_g_int, dphi_g_ext))
        
        # Centrifugal potential 
        i_est = np.abs(dr) > EPS
        phi_c_est, dphi_c_est = find_centrifugal_potential(
            r_est[i_est], cth, omega
            )
        
        # Total potential 
        phi_t_est  =  phi_g_est +  phi_c_est
        dphi_t_est = dphi_g_est + dphi_c_est
        
        # Update r_est
        dr[i_est] = -(phi_t_est - target[i_est]) / dphi_t_est
        r_est[i_est] += dr[i_est]
    return r_est        

def find_new_mapping(map_n, omega_n, phi_g_l, phi_eff, cth) :
    """
    Find the new mapping by comparing the effective potential
    and the total potential (calculated from phi_g_l and omega_n).

    Parameters
    ----------
    map_n : ARRAY(N, M)
        Current mapping.
    omega_n : FLOAT
        Current rotation rate.
    phi_g_l : ARRAY(N, L)
        Gravitation potential harmonics.
    phi_eff : ARRAY(N, )
        Effective potential on each equipotential.
    cth : ARRAY(M, )
        Value of cos(theta)in the Gauss-Legendre scheme.

    Returns
    -------
    map_n_new : ARRAY(N, M)
        Updated mapping.
    omega_n_new : FLOAT
        Updated rotation rate.

    """
    # 2D gravitational potential
    phi2D_g = pl_eval_2D(phi_g_l, cth)
    
    # Find a new value for ROT
    k_eq = (M-1)//2
    omega_n_new = estimate_omega(
        map_n[:, k_eq], phi_eff, phi2D_g[:, k_eq], phi_g_l[-1], 0.0, omega_n
        )
    
    # Finding the new mapping
    map_n_new = np.copy(map_n)
    all_k = np.arange(k_eq+1)
    find_mapping_k = lambda k : find_mapping_theta(
        map_n[:, k], phi_eff, phi2D_g[:, k], phi_g_l[-1], cth[k], omega_n_new
        )
    map_n_new[:, all_k+0] = np.array([find_mapping_k(k) for k in all_k]).T
    map_n_new[:,-all_k-1] = map_n_new[:, all_k]
        
    return map_n_new, omega_n_new


def plot_mapping(map_n, phi_eff, cth, weights, 
                 n_lines=50, cmap=cm.hot_r, size=16) :
    """
    Plot the equipotentials of a given mapping (map_n).

    Parameters
    ----------
    map_n : ARRAY(N, M)
        2D Mapping.
    phi_eff : ARRAY(N, )
        Value of the effective potential on each equipotential.
        Serves the colormapping.
    cth : ARRAY(M, )
        Value of cos(theta)in the Gauss-Legendre scheme.
    weights : ARRAY(M, )
        Integration weights in the Gauss-Legendre scheme.
    n_lines : INTEGER, optional
        Number of equipotentials on the plot. The default is 50.
    cmap : cm.cmap instance, optional
        Colormap for the plot. The default is cm.viridis_r.
    size : INTEGER, optional
        Fontsize. The default is 16.

    Returns
    -------
    None.

    """
    
    # Angular interpolation
    map_l   = pl_project_2D(map_n, cth, weights)
    map_res = pl_eval_2D(map_l, np.linspace(-1, 1, RES))
    cth_res = np.linspace(-1, 1, RES)
    sth_res = np.sqrt(1-cth_res**2)
    
    # Right side
    rs = LineCollection(
        [np.column_stack([x, y]) for x, y in zip(
            map_res[::-N//n_lines]*sth_res, 
            map_res[::-N//n_lines]*cth_res
            )], 
        cmap=cmap, 
        linewidths=1.0
        )
    
    #Left side
    ls = LineCollection(
        [np.column_stack([x, y]) for x, y in zip(
            -map_res[::-N//n_lines]*sth_res, 
             map_res[::-N//n_lines]*cth_res
            )], 
        cmap=cmap, 
        linewidths=1.0
        )
    
    
    # Text formating 
    rc('text', usetex=True)
    rc('xtick', labelsize=size)
    rc('ytick', labelsize=size)
    
    # Plot
    rs.set_array(phi_eff[::-N//n_lines])
    ls.set_array(phi_eff[::-N//n_lines])
    plt.close('all')
    fig, ax = plt.subplots()
    ax.add_collection(rs)
    ax.add_collection(ls)
    axcb = fig.colorbar(rs)
    axcb.ax.set_title(
        r"$\phi_\mathrm{eff}(\zeta)$", 
        y=1.03, fontsize=size+3
        )
    plt.axis('equal')
        
        
    
    
#%% Sph functions cell

def find_metric_terms(map_n) : 
    """
    Finds the metric terms, i.e the derivatives of r(z, t) 
    with respect to z or t (with z := zeta and t := cos(theta)).

    Parameters
    ----------
    map_n : ARRAY(N, M)
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
    map_l = pl_project_2D(dr._, cth, weights)
    _, dr.t, dr.tt = pl_eval_2D(map_l, cth, der=2)
    dr.z = np.array(
        [interpolate_func(
            zeta, rk, der=1, k=SPL_ORDER
            )(zeta) for rk in map_n.T]
        ).T
    map_l_z = pl_project_2D(dr.z, cth, weights)
    _, dr.zt, dr.ztt = pl_eval_2D(map_l_z, cth, der=2)
    return dr


def Legendre_coupling(f, t, weights, der=(0, 0)) :
    """
    Finds the harmonic couplings of a given f function, that is:
        \mathcal{P}^{\ell\ell'}_f(\zeta) = 
    \int_{-1}^1 f(\zeta, t) P_\ell^{(d_\ell)}(t)P_{\ell'}^{(d_{\ell'})}(t)\,dt

    with P_\ell the l-th Legendre polynomial and d_\ell a derivative order.
    
    Parameters
    ----------
    f : ARRAY(..., M)
        Input function discretised on the mapping.
    t : ARRAY(M)
        Values of cos(theta).
    weights : ARRAY(M)
        Gauss-Legendre scheme weights.
    der : tuple of INT, optional
        Derivative orders for the Legendre polynomials. 
        The default is (0, 0).

    Returns
    -------
    Pll : ARRAY(..., L, L)
        Harmonic couplings of f.

    """
    # pl computation
    pl_series = orthopy.c1.legendre.Eval(t, "normal")
    all_pl   = np.array([next(pl_series) for l in range(L)])
    all_dpl  = np.empty_like(all_pl)
    all_ddpl = np.empty_like(all_pl)
    
    if np.any(np.array(der) >= 1) :
        # dpl computation
        all_l = np.arange(L)
        norm = np.sqrt(2 / (2*all_l+1))
        all_dpl = all_l[:, None] * np.roll(norm[:, None] * all_pl, 1, axis=0)
        for l in range(1, L) : 
            all_dpl[l] += t * all_dpl[l-1]
        all_dpl /= norm[:, None]
        
        if np.any(np.array(der) >= 2) :
            # ddpl computation
            all_lp1 = np.where(all_l != 0, all_l+1, 0)
            all_ddpl = all_lp1[:, None] * np.roll(
                norm[:, None] * all_dpl, 1, axis=0
                )
            for l in range(1, L) : 
                all_ddpl[l] += t * all_ddpl[l-1]
            all_ddpl /= norm[:, None]
            
    pl1, pl2 = np.choose(
        np.array(der)[:, None, None], 
        choices=[all_pl[::2], all_dpl[::2], all_ddpl[::2]]
        )
    
    Pll = np.einsum(
        '...k,lk,mk->...lm', 
        weights * np.atleast_2d(f), pl1, pl2, 
        optimize='optimal'
        )
    return Pll


def find_all_couplings(dr, t, weights) :
    """
    Find all the couplings needed to solve Poisson's equation in 
    spheroidal coordinates.

    Parameters
    ----------
    dr : DotDict instance
        The mapping and its derivatives with respect to z and t
    t : ARRAY(M)
        Values of cos(theta).
    weights : ARRAY(M)
        Gauss-Legendre scheme weights.

    Returns
    -------
    Pll : DotDict instance
        Harmonic couplings. They are caracterised by their related 
        metric term : {
            zz : ARRAY(N, Nl, Nl)
                coupling associated to phi_zz,
            zt : ARRAY(N, Nl, Nl)
                            //         phi_zt,
            tt : ARRAY(N, Nl, Nl)
                            //         phi_tt
            }

    """
    Pll = DotDict()
    l = np.arange(0, L, 2)
    
    Pll.zz = Legendre_coupling(
        (dr._**2 + (1-t**2) * dr.t**2) / dr.z, t, weights, der=(0, 0)
        )
    Pll.zt = Legendre_coupling(
        (1-t**2) * dr.tt - 2*t * dr.t, t, weights, der=(0, 0)
        ) + 2 * Legendre_coupling(
        (1-t**2) * dr.t, t, weights, der=(0, 1)
        )
    Pll.tt = Legendre_coupling(
        dr.z, t, weights, der=(0, 0)
        ) * l*(l+1)
    
    return Pll

def find_outer_BC(dr, t, weights) : 
    """
    Finds the adequate matrix to ensure the outer boundary conditions. 

    Parameters
    ----------
    dr : DotDict instance
        The mapping and its derivatives with respect to z and t
    t : ARRAY(M)
        Values of cos(theta).
    weights : ARRAY(M)
        Gauss-Legendre scheme weights.

    Returns
    -------
    BC0, BC1 : ARRAY(Nl, Nl), ARRAY(Nl, Nl)
        Outer boundary conditions applied to the potential harmonics.

    """
    pl_series = orthopy.c1.legendre.Eval(t, "normal")
    all_pl   = np.array([next(pl_series) for l in range(L)])
    
    Nl = (L+1)//2
    surf, dsurf = dr._[-1], dr.z[-1]
    surf_01 = np.array([(l+1) * dsurf*surf**(-l-2) for l in range(0, L, 2)])
    surf_10 = np.array([surf**(Nl    ) for l in range(0, L, 2)])
    surf_11 = np.array([surf**(Nl-l-1) for l in range(0, L, 2)])
    
    pl = all_pl[::2]
    P01 = pl @ (weights * surf_01 * pl).T
    P10 = pl @ (weights * surf_10 * pl).T
    P11 = pl @ (weights * surf_11 * pl).T
    BC0 = np.eye(Nl)
    BC1 = P01 @ np.linalg.inv(P11) @ P10 
    # print(f"Cond(BC1) = {np.linalg.cond(BC1)}")
    return BC0, BC1


    
    

#%% Main cell

if __name__ == '__main__' :
    
    # Definition of global parameters
    NAME_1D, ROT, FULL, eval_phi_c, ALPHA, SCALE, EPS, SPL_ORDER, \
    LAG_ORDER, L, M, RES         = set_params() 
        
    # Definition of the 1D-model
    P0, N, r1D, rho1D            = read_1D()  
    
    # Non-dimensional quantities               
    mass, radius, r, zeta, rho_n = renormalise_1D(r1D, rho1D)
    
    # Angular domain preparation
    cth, weights, map_n          = init_2D()
    
    # Find the lagrange matrix
    lag = lagrange_matrix(zeta, order=LAG_ORDER)
    
    # Define sparse matrices 
    Lsp = sps.dia_matrix(lag.mat[..., 0])
    Dsp = sps.dia_matrix(lag.mat[..., 1])
    
    # Initialisation for the effective potential
    _, phi_eff, dphi_eff = find_phi_eff(
        map_n, rho_n, 0.0, cth, weights
        )
    
    # Iterative centrifugal deformation
    surfaces = [map_n[-1]]
    r_pol = [0.0, map_n[-1, -1]]
    omega = [0.0]
    n = 1
    while abs(r_pol[-1] - r_pol[-2]) > EPS :
        
        # Current rotation rate
        omega_n = min(ROT, (n/FULL) * ROT)
        
        # Effective potential computation
        phi_g_l, phi_eff = find_phi_eff(
            map_n, rho_n, omega_n, cth, weights, phi_eff
            )
        
        # Update the mapping
        map_n, omega_n = find_new_mapping(
            map_n, omega_n, phi_g_l, phi_eff, cth
            )        

        # Renormalisation
        r_corr    = find_r_eq(map_n, cth, weights)
        m_corr    = find_mass(map_n, rho_n, weights)   
        radius   *= r_corr
        mass     *= m_corr
        map_n    /= r_corr
        rho_n    /= m_corr/r_corr**3
        phi_eff  /= m_corr/r_corr
        dphi_eff /= m_corr/r_corr**3
        
        # Update the surface and polar radius
        surfaces.append(map_n[-1])
        r_pol.append(find_r_pol(map_n, cth, weights))
        omega.append(omega_n)
        
        # Iteration count
        print(f"Iteration n°{n}, R_pol = {r_pol[-1].round(12)},",
              f"Omega = {omega_n.round(12)}")
        n += 1
    
    # Find pressure
    P = find_pressure(rho_n, dphi_eff)
    
    # Interpolation
    plot_mapping(map_n, phi_eff, cth, weights)
    
        
    
    
    
    

    