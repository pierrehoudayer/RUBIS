#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 12:14:01 2022

@author: phoudayer
"""

import matplotlib.pyplot as plt
import numpy             as np
import probnum           as pn
from numpy.polynomial.polynomial import Polynomial
from itertools                   import combinations
from matplotlib                  import rc
from matplotlib.collections      import LineCollection
from pylab                       import cm
from scipy.interpolate           import splrep, splantider, splev, splint
from scipy.special               import expn, eval_legendre, roots_legendre


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

def test(x, y) :
    return pn.quad.bayesquad_from_data(x, y)
    

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
    
    
def pl_project_2D(f, L) :
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
    f_l = norm * np.array(
        [project(f, l) if (l%2 == 0) else zeros(f) for l in range(L)]
    ).T
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

def plot_f_map(
    map_n, f, phi_eff, max_degree,
    angular_res=501, levels=100, cmap=cm.Blues, size=16, label=r"$f$",
    show_surfaces=False, n_lines=50, cmap_lines=cm.BuPu, lw=0.5,
    disc=None, map_ext=None, n_lines_ext=20
) :
    """
    Shows the value of f in the 2D model.

    Parameters
    ----------
    map_n : array_like, shape (N, M)
        2D Mapping.
    f : array_like, shape (N, )
        Function value on the surface levels.
    phi_eff : array_like, shape (N, )
        Value of the effective potential on each isopotential.
        Serves the colormapping if show_surfaces=True.
    max_degree : integer
        number of harmonics to use for interpolating the mapping.
    angular_res : integer, optional
        angular resolution used to plot the mapping. The default is 501.
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

    Returns
    -------
    None.

    """
    
    # Angular interpolation
    N, _ = map_n.shape
    cth_res = np.linspace(-1, 1, angular_res)
    sth_res = np.sqrt(1-cth_res**2)
    map_l   = pl_project_2D(map_n, max_degree)
    map_res = pl_eval_2D(map_l, np.linspace(-1, 1, angular_res))
    
    # 2D density
    f2D = np.tile(f, angular_res).reshape((angular_res, N)).T
    
    # Text formating 
    rc('text', usetex=True)
    rc('xtick', labelsize=size)
    rc('ytick', labelsize=size)
    
    # Init figure
    fig, ax = plt.subplots(figsize=(15, 8.4), frameon=False)
    
    # Right side
    csr = ax.contourf(
        map_res*sth_res, map_res*cth_res, f2D, 
        cmap=cmap, levels=levels
    )
    for c in csr.collections:
        c.set_edgecolor("face")
    if disc is not None :
        for i in disc :
            plt.plot(map_res[i]*sth_res, map_res[i]*cth_res, 'w-', lw=lw)
    plt.plot(map_res[-1]*sth_res, map_res[-1]*cth_res, 'k--', lw=lw)
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
            cmap=cmap, levels=levels
        )
        for c in csl.collections:
            c.set_edgecolor("face")
        if disc is not None :
            for i in disc :
                plt.plot(-map_res[i]*sth_res, map_res[i]*cth_res, 'w-', lw=lw)
        plt.plot(-map_res[-1]*sth_res, map_res[-1]*cth_res, 'k--', lw=lw)
        
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
