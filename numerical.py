#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 12:14:01 2022

@author: phoudayer
"""

import numpy as np
from numpy.polynomial.polynomial import Polynomial
from itertools                   import combinations
from scipy.interpolate           import splrep, splantider, splev, splint
from scipy.special               import expn, roots_legendre

def lnxn(x, n=1, a=1.) : 
    """
    Function returning the value of: y(x) = x^a \ln^n(x) and continuates it
    in 0 by y(0) = 0.

    Parameters
    ----------
    x : float or array_like
        Input value
    n : INT, optional
        Logarithm exponent value. The default is 1.
    a : float, optional
        Polynamial exponent value (can be real). The default is 1.0.

    Returns
    -------
    y : float or array_like (same shape as x)
        Output value

    """
    with np.errstate(all='ignore') :
        y = np.where(x == 0, 0.0, np.log(x)**n * x**a)
        return y

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
    x : float or array_like
        Input value
    k : float, optional
        Exponent value, it can theoritically be a real but must
        be an integer if one want to compute the analytical
        primitive of the function (multivalued otherwise). 
        The default is 1.
    a : float, optional
        Add an optional homotesy to the axis. The default is 1.

    Returns
    -------
    y : float or array_like (same shape as x)
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
    x : float or array_like
        Input value
    k : INT, optional
        Argument of expinv(x, k). The default is 1.
    a : float, optional
        Add an optional homotesy to the axis. The default is 1.

    Returns
    -------
    y : float or array_like (same shape as x)
        Output value

    """
    with np.errstate(all='ignore') :
        u = x**-(1/k)
        y = k * expn(k+1, a*u)
        return y
    
def del_u_over_v(du, dv, der) : 
    """
    Computes the derivatives of u/v

    Parameters
    ----------
    du : list of float or array_like
        derivatives of u.
    dv : list of float or array_like
        derivatives of v.
    der : INT in {0, 1, 2}
        Derivative order.

    Returns
    -------
    y : float or array_like 
        Output value

    """
    assert der in {0, 1, 2}
    if der == 0 : 
        y = du[0] / dv[0]
    elif der == 1 : 
        y = (
            + du[1]       / dv[0] 
            - du[0]*dv[1] / dv[0]**2
            )
    else : 
        y = (
            +    du[2]                      / dv[0] 
            - (2*du[1]*dv[1] + du[0]*dv[2]) / dv[0]**2 
            +  2*du[0]*dv[1]**2             / dv[0]**3
            )
    return y 
   

def integrate(x, y, a=None, b=None, k=3) : 
    """
    Function computing the integral of f(x) between a and b
    for fixed sampled values of y_i = f(x_i) at x_i. The routine 
    makes use of the scipy.interpolate.splXXX functions to perform
    this integral using B-splines.

    Parameters
    ----------
    x : array_like, shape (N, )
        x values on which to integrate.
    y : array_like, shape (N, )
        y values to integrate.
    a, b : floats, optional
        Lower and upper bounds used to compute the integral. 
        The default is None.
    k : INT, optional
        Degree of the B-splines used to compute the integral. 
        The default is 3.

    Returns
    -------
    integral : float
        Result of the integration.

    """
    tck = splrep(x, y, k=k)
    if a == None :
        a = x[0]
    if b == None : 
        b = x[-1]
    integral = splint(a, b, tck)
    return integral

def integrate2D(r2D, y, domains=None, k=3) : 
    """
    Function computing the integral of f(r, cos(theta)) for fixed 
    sampled values of y_ij = f(r2D_ij, mu_j) with mu_j the values
    of cos(theta) evaluated on the Gausse-Legendre nodes. Here r2D
    depends is generally assumed to be 2 dimensional, allowing the
    grid to be dependant on the angle.

    Parameters
    ----------
    r2D : array_like, shape (N, M)
        radial values (along the 1st axis) as a function of
        cos(theta) (along the 2nd axis).
    y : [array_like, shape (N, ) or (N, M)] or [callable(r, cth, D)]
        if type == array :
            y values to integrate and evaluated on r2D. If y is
            dimensional, it will be assumed to be independent of
            the angle.
        if callable(y) :
            function of r and cos(theta) for each domain (D) 
            of r2D.
    domains : tuple of arrays
        If given, the different domains on which the integration
        should be carried independently. Useful if y happens to
        be discontinuous on given r2D[i] lines. 
        The default is None.
    k : INT, optional
        Degree of the B-splines used to compute the integral. 
        The default is 3.

    Returns
    -------
    integral : float
        Result of the integration.

    """
    # Definition of Gauss-Legendre nodes
    N, M = r2D.shape
    cth, weights = roots_legendre(M)
    
    # Integration in given angular directions
    if domains is None: domains = (np.arange(N), )
    if callable(y) :
        radial_integral = np.array([sum(
            integrate(rk[D], y(rk, ck, D) * rk[D]**2, k=5) for D in domains
        ) for rk, ck in zip(r2D.T, cth)])
    else :
        if len(y.shape) < 2 : y = np.tile(y, (M, 1)).T
        radial_integral = np.array([sum(
            integrate(rk[D], yk[D] * rk[D]**2, k=5) for D in domains
        ) for rk, yk in zip(r2D.T, y.T)])
    
    # Integration over the angular domain
    integral = 2*np.pi * radial_integral @ weights
    return integral

def interpolate_func(x, y, der=0, k=3, prim_cond=None, *args, **kwargs):
    """
    Routine returning an interpolation function of (x, y) 
    for a given B-spline order k. A derivative order can 
    be specified by the value der < k 
    (if der=-1, returns an antiderivative).

    Parameters
    ----------
    x : array_like, shape (N, )
        x values on which to integrate.
    y : array_like, shape (N, )
        y values to integrate.
    der : INT, optional
        Order of the derivative. 
        The default is 0.
    k : INT, optional
        Degree of the B-splines used to compute the integral. 
        The default is 3.
    s : float, optional
        Smoothing parameter. 
        The default is 0.
    prim_cond : array_like, shape (2, ), optional
        Conditions to specify the constant to add to the
        primitive function if der = -1. The first value 
        is an integer i, such that F(x[i]) = second value.
        The default is None, which correspond to F(x[0]) = 0.

    Returns
    -------
    func : function(x_eval)
        Interpolation function of x_eval.

    """
    tck = splrep(x, y, k=k, *args, **kwargs)
    if not type(der) is int :
        def func(x_eval) :
            if 0 not in np.asarray(x_eval).shape :
                return [splev(x_eval, tck, der=d) for d in der]
            else : 
                return np.array([]).reshape((len(der), 0))
        return func
    if not -2 < der < k : 
        raise ValueError(
            f"""Derivative order should be either -1 (antiderivative)
            or 0 <= der < k={k} (derivative). Current value is {der}."""
            )
    if der >= 0 :
        def func(x_eval) :
            if 0 not in np.asarray(x_eval).shape :
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
    
def find_roots(c):
    """
    Vectorial solver of cubic and quadratic equations with real roots
    and coefficients. Return the only root we want in lagrange_matrix
    and therefore should not be used in any other way.
    
    Parameters
    ----------
    c : array_like, shape (3, N-1)
        The coefficients of the cubic equation.
    
    Returns
    -------
    roots : array_like, shape (N-1, )
        One root per cubic equation.
    """    
    solve_cubic = np.ones(c.shape[1], dtype='bool')
    solve_cubic[0], solve_cubic[-1] = False, False
    roots = np.empty((c.shape[1], ))
    
    c2, c3 = c[:, ~solve_cubic], c[:, solve_cubic]

    D = np.sqrt(c2[1] * c2[1] - 4.0 * c2[2] * c2[0])
    roots[ 0] = (-c2[1, 0] - D[0]) / (2.0 * c2[2, 0])
    roots[-1] = (-c2[1, 1] + D[1]) / (2.0 * c2[2, 1])
            
    f = (  (3.0 * c3[1] / c3[3]) 
         - ((c3[2] ** 2.0) / (c3[3] ** 2.0))       ) / 3.0                         
    g = ( ((2.0 * (c3[2] ** 3.0)) / (c3[3] ** 3.0)) 
         - ((9.0 * c3[2] * c3[1]) / (c3[3] ** 2.0)) 
         + (27.0 * c3[0] / c3[3])                  ) /27.0                    
    h = ((g ** 2.0) / 4.0 + (f ** 3.0) / 27.0)                          

    i = np.sqrt(((g ** 2.0) / 4.0) - h)  
    j = i ** (1 / 3.0)                     
    k = np.arccos(-(g / (2 * i)))          
    L = j * -1                             
    M = np.cos(k / 3.0)                  
    N = np.sqrt(3) * np.sin(k / 3.0)   
    P = (c3[2] / (3.0 * c3[3])) * -1      
    
    roots[solve_cubic] = L * (M - N) + P
    
    return roots

def find_root_i(i, t_i, c_i, order) : 
    """
    Finds the root of the polynomial with coefficients c_i that lies
    between the adequate values of t_i (which depends on the index i).
    
    Parameters
    ----------
    i : integer
        Polynomial index
    t_i : array_like, shape ([order+1, 2*order], ) (depends on i)
        Window values
    c_i : array_like, shape (2*order, )
        Coefficients of the polynomial
    order : INT
        Scheme order. The latter determines the degree of the polynomial
        which is 2*order-1.
        
    Returns
    -------
    root_i : float
        Adequate polynomial root
    """
    deriv_roots = Polynomial(c_i).roots()
    lb = t_i[order-1 - max(order-1 - i, 0)] 
    ub = t_i[order-0 - max(order-1 - i, 0)]
    root_i = float(deriv_roots[(lb < deriv_roots) & (deriv_roots < ub)])
    return root_i

def lagrange_matrix_P(x, order=2) :
    """
    Computes the interpolation and derivation matrices based on
    an initial grid x. The new grid on which the 
    interpolation/derivation takes place is entierly defined 
    from the x nodes following Reese (2013). The use of
    'order=2' is recommended since the roots of the polynomials
    involved in the routine (degree = 2*order-1) can be found
    analytically, resulting in a faster determination of mat.
    
    Parameters
    ----------
    x : array_like, shape (N, )
        Initial grid from which one interpolates/derives
    order : integer, optional
        Scheme order from the lagrange interpolation/derivation.
        The effective precision order is 2*order, even though the 
        number of points involved in each window is also 2*order.
        The default is 2.

    Returns
    -------
    mat : array_like, shape (N-1, N, 2)
        Contains the interpolation (mat[...,0]) and
        the derivation (mat[...,1]) matrices.

    """
    N = len(x)
    mat  = np.zeros((N-1, N, 2))
    
    convolve_same = lambda a, v: np.convolve(a, v, mode='same')
    vconv = np.vectorize(convolve_same, signature='(n),(m)->(n)')
    mask = vconv(np.eye(N, dtype='bool'), [True]*2*order)[:-1]   
    
    rescale = lambda x, ref : (2*x - (ref[-1] + ref[0]))/ (ref[-1] - ref[0])
    
    t = [rescale(x[mask_i], x[mask_i]) for mask_i in mask]
    s = np.array([0.5*(x[mask_i][-1] - x[mask_i][0]) for mask_i in mask])
    coefs = lambda t : (
          [sum(np.prod(list(combinations(t, k)), axis=1))*(len(t)-k)*(-1)**k for k in range(len(t)-1, -1, -1)] 
        + [0.0]*(2*order-len(t))
        )
    c = np.array(list(map(coefs, t)))
    
    if order == 2 :
        roots = find_roots(c.T)
    else : 
        roots = [find_root_i(i, t_i, c_i, order) for i, (t_i, c_i) in enumerate(zip(t, c))]
        
    for i, (t_i, root_i, mask_i) in enumerate(zip(t, roots, mask)) :
        n_i  = len(t_i)
        t_ij = np.tile(t_i, n_i)[([False] + [True]*n_i)*(n_i-1) + [False]].reshape((n_i, -1))
        
        l_i = np.prod((root_i - t_ij)/(t_i[:, None] - t_ij), axis=1)
        d_i = np.sum(l_i[:, None] / (root_i - t_ij), axis=1)
    
        mat[i, mask_i, 0] = l_i
        mat[i, mask_i, 1] = d_i
    mat[..., 1] /= s[:, None]

    return mat
    
# def lagrange_matrix_F(x, order=2) :
#     """
#     Computes the interpolation and derivation matrices based on
#     an initial grid x. The new grid, on which the 
#     interpolation/derivation takes place, is entierly defined 
#     from the x nodes following Reese (2013). Makes use of 
#     a Fortran routine.

#     Parameters
#     ----------
#     x : array_like, shape (N, )
#         Initial grid from which one interpolates/derives
#     order : INT, optional
#         Scheme order from the lagrange interpolation/derivation.
#         The effective precision order is 2*order, even though the 
#         number of points involved in each window is also 2*order.
#         The default is 2.

#     Returns
#     -------
#     mat : array_like, shape(N-1, N, 2)
#         Contains the interpolation (mat[...,0]) and
#         the derivation (mat[...,1]) matrices.

#     """
#     from init_derive_IFD import init_derive_ifd
#     mat_lag, _ = init_derive_ifd(x, 1, order)
#     return mat_lag