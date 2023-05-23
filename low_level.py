#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 12:14:01 2022

@author: phoudayer
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcl
import numpy             as np
from numpy.polynomial.polynomial import Polynomial
from itertools                   import combinations
from matplotlib                  import rc
from matplotlib.collections      import LineCollection
from pylab                       import cm
from scipy.interpolate           import splrep, splantider, splev, splint
from scipy.special               import expn, eval_legendre, roots_legendre

from dotdict                     import DotDict


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
    
def find_r_eq(map_n, L) :
    """
    Function to find the equatorial radius from the mapping.

    Parameters
    ----------
    map_n : array_like, shape (N, M)
        Isopotential mapping.
    L : integer
        truncation order for the harmonic series expansion.

    Returns
    -------
    r_eq : float
        Equatorial radius.

    """
    surf_l = pl_project_2D(map_n[-1], L)
    return pl_eval_2D(surf_l, 0.0)

def find_r_pol(map_n, L) :
    """
    Function to find the polar radius from the mapping.

    Parameters
    ----------
    map_n : array_like, shape (N, M)
        Isopotential mapping.
    L : integer
        truncation order for the harmonic series expansion.

    Returns
    -------
    r_eq : float
        Equatorial radius.

    """
    surf_l = pl_project_2D(map_n[-1], L)
    return pl_eval_2D(surf_l, 1.0)
    
    
def pl_project_2D(f, L, even=True) :
    """
    Projection of function, assumed to be already evaluated 
    at the Gauss-Legendre scheme points, over the Legendre 
    polynomials.    

    Parameters
    ----------
    f : array_like, shape (N, M)
        function to project.
    L : integer
        truncation order for the harmonic series expansion.
    even : boolean, optional
        should the function assume that f is even?

    Returns
    -------
    f_l : array_like, shape (N, L)
        The projection of f over the legendre polynomials
        for each radial value.

    """
    N, M = np.atleast_2d(f).shape
    cth, weights = roots_legendre(M)
    zeros = lambda f: np.squeeze(np.zeros((N, )))
    project = lambda f, l: f @ (weights * eval_legendre(l, cth))
    norm = (2*np.arange(L)+1)/2
    if even :
        f_l = norm * np.array(
            [project(f, l) if (l%2 == 0) else zeros(f) for l in range(L)]
        ).T
    else : 
        f_l = norm * np.array([project(f, l) for l in range(L)]).T
    return f_l


def pl_eval_2D(f_l, t, der=0) :
    """
    Evaluation of f(r, t) (and its derivatives) from a projection,
    f_l(r, l), of f over the Legendre polynomials.

    Parameters
    ----------
    f_l : array_like, shape (N, L)
        The projection of f over the legendre polynomials.
    t : array_like, shape (N_t, )
        The points on which to evaluate f.
    der : integer in {0, 1, 2}
        The upper derivative order. The default value is 0.
    Returns
    -------
    f : array_like, shape (N, N_t)
        The evaluation of f over t.
    df : array_like, shape (N, N_t), optional
        The evaluation of the derivative f over t.
    d2f : array_like, shape (N, N_t), optional
        The evaluation of the 2nd derivative of f over t.

    """
    assert der in {0, 1, 2} # Check the der input
    
    # f computation
    _, L = np.atleast_2d(f_l).shape
    pl = np.array([eval_legendre(l, t) for l in range(L)])
    f = f_l @ pl
    
    if der != 0 :
        # df computation
        ll = np.arange(L)[:, None]
        dpl = ll * np.roll(pl, 1, axis=0)
        for l in range(1, L):
            dpl[l] += t * dpl[l-1]
        df = f_l @ dpl
        
        if der != 1 :
            # d2f computation
            llp1 = np.where(ll != 0, ll+1, 0)
            d2pl = llp1 * np.roll(dpl, 1, axis=0)
            for l in range(1, L):
                d2pl[l] += t * d2pl[l-1]
            d2f = f_l @ d2pl
            
            return f, df, d2f
        return f, df
    return f

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

def Legendre_coupling(f, L, der=(0, 0)) :
    """
    Finds the harmonic couplings of a given f function, that is:
        \mathcal{P}^{\ell\ell'}_f(\zeta) = 
    \int_{-1}^1 f(\zeta, t) P_\ell^{(d_\ell)}(t)P_{\ell'}^{(d_{\ell'})}(t)\,dt

    with P_\ell the l-th Legendre polynomial and d_\ell a derivative order.
    
    Parameters
    ----------
    f : array_like, shape (..., M)
        Input function discretised on the mapping.
    L : integer
        Highest harmonic degree.
    der : tuple of integer, optional
        Derivative orders for the Legendre polynomials. 
        The default is (0, 0).

    Returns
    -------
    Pll : array_like, shape (..., L, L)
        Harmonic couplings of f.

    """    
    # Gauss-Legendre scheme
    *_, M = f.shape
    cth, weights = roots_legendre(M)
    
    # pl computation
    pl   = np.array([eval_legendre(l, cth) for l in range(L)])
    dpl  = np.empty_like(pl)
    d2pl = np.empty_like(pl)
    
    if der != 0 :
        # dpl computation
        ll = np.arange(L)[:, None]
        dpl = ll * np.roll(pl, 1, axis=0)
        for l in range(1, L):
            dpl[l] += cth * dpl[l-1]
        
        if der != 1 :
            # d2pl computation
            llp1 = np.where(ll != 0, ll+1, 0)
            d2pl = llp1 * np.roll(dpl, 1, axis=0)
            for l in range(1, L):
                d2pl[l] += cth * d2pl[l-1]
            
    pl1, pl2 = np.choose(
        np.array(der)[:, None, None], 
        choices=[pl[::2], dpl[::2], d2pl[::2]]
    )
    
    Pll = np.einsum(
        '...k,lk,mk->...lm', 
        weights * np.atleast_2d(f), pl1, pl2, 
        optimize='optimal'
    )
    return Pll
    
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
    
def lagrange_matrix_F(x, order=2) :
    """
    Computes the interpolation and derivation matrices based on
    an initial grid x. The new grid, on which the 
    interpolation/derivation takes place, is entierly defined 
    from the x nodes following Reese (2013). Makes use of 
    a Fortran routine.

    Parameters
    ----------
    x : array_like, shape (N, )
        Initial grid from which one interpolates/derives
    order : INT, optional
        Scheme order from the lagrange interpolation/derivation.
        The effective precision order is 2*order, even though the 
        number of points involved in each window is also 2*order.
        The default is 2.

    Returns
    -------
    mat : array_like, shape(N-1, N, 2)
        Contains the interpolation (mat[...,0]) and
        the derivation (mat[...,1]) matrices.

    """
    from init_derive_IFD import init_derive_ifd
    mat_lag, _ = init_derive_ifd(x, 1, order)
    return mat_lag


def app_list(val, idx, func=lambda x: x, args=None) :
    """
    Function only designed for convenience in the vectorial mapping finding.

    Parameters
    ----------
    val : list
        Values on which func is applied
    idx : list
        val ordering.
    func : function or list of function, optional
        functions to be applied on val. The default is lambda x: x (identity).
    args : list, optional
        list of function arguments. The default is None.

    Returns
    -------
    array_like
        The function applied to val with corresponding args.

    """
    unq_idx = np.unique(idx)
    if (args is None) & (callable(func)) :
        Func = lambda l: func(val[idx == l])
    elif (args is None) & (not callable(func)) :
        Func = lambda l: func[l](val[idx == l])
    elif (args is not None) & (callable(func)) :
        Func = lambda l: func(val[idx == l], args[l])
    else :
        Func = lambda l: func[l](val[idx == l], args[l])
    return np.hstack(list(map(Func, unq_idx)))

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

def plot_f_map(
    map_n, f, phi_eff, max_degree,
    angular_res=501, t_deriv=0, levels=100, cmap=cm.Blues, size=16, label=r"$f$",
    show_surfaces=False, n_lines=50, cmap_lines=cm.BuPu, lw=0.5,
    disc=None, map_ext=None, n_lines_ext=20,
    add_to_fig=None
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
    disc : array_like, shape (Nd, )
        Indices of discontinuities to plot. The default is None.
    map_ext : array_like, shape (Ne, M), optional
        Used to show the external mapping, if given.
    n_lines_ext : integer, optional
        Number of level surfaces in the external mapping. The default is 20.
    add_to_fig : fig object, optional
        If given, the figure on which the plot should be added. 
        The default is None.

    Returns
    -------
    None.

    """
    
    # Angular interpolation
    N, _ = map_n.shape
    cth_res = np.linspace(-1, 1, angular_res)
    sth_res = np.sqrt(1-cth_res**2)
    map_l   = pl_project_2D(map_n, max_degree)
    map_res = pl_eval_2D(map_l, cth_res)
    
    # 2D density
    if len(f.shape) == 1 :
        f2D = np.tile(f, angular_res).reshape((angular_res, N)).T
    else : 
        f_l = pl_project_2D(f, max_degree, even=False)
        f2D =np.atleast_3d(np.array(pl_eval_2D(f_l, cth_res, der=t_deriv)).T).T[-1]
        
    # Text formating 
    rc('text', usetex=True)
    rc('xtick', labelsize=size)
    rc('ytick', labelsize=size)
    # rc('axes', facecolor='#303030')
    
    # Init figure
    if add_to_fig is None : 
        fig, ax = plt.subplots(figsize=(15, 8.4), frameon=False)
    else : 
        fig, ax = add_to_fig
    norm = None
    if (cmap is cm.Blues)&(np.nanmin(f2D) * np.nanmax(f2D) < 0.0) : 
        cmap, norm = cm.RdBu_r, mcl.CenteredNorm()
    
    # Right side
    csr = ax.contourf(
        map_res*sth_res, map_res*cth_res, f2D, 
        cmap=cmap, norm=norm, levels=levels
    )
    for c in csr.collections:
        c.set_edgecolor("face")
    if disc is not None :
        for i in disc :
            plt.plot(map_res[i]*sth_res, map_res[i]*cth_res, 'w-', lw=lw)
    plt.plot(map_res[-1]*sth_res, map_res[-1]*cth_res, 'k-', lw=lw)
    cbr = fig.colorbar(csr, aspect=30)
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
        cbl = fig.colorbar(ls, location='left', pad=0.15, aspect=30)
        cbl.ax.set_title(
            r"$\phi_\mathrm{eff}(\zeta)$", 
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
    plt.axis('equal')
    plt.xlim((-1, 1))
    plt.xlabel('$s/R_\mathrm{eq}$', fontsize=size+3)
    plt.ylabel('$z/R_\mathrm{eq}$', fontsize=size+3)
    plt.show()
