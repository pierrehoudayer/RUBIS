#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 12:14:01 2022

@author: phoudayer
"""

import numpy as np
from scipy.interpolate           import splrep, splantider, splev, splint
from scipy.special               import expn

def lnxn(x, n=1, a=1.) : 
    """
    Function returning the value of: y(x) = x^a \ln^n(x) and continuates it
    in 0 by y(0) = 0.

    Parameters
    ----------
    x : FLOAT or ARRAY
        Input value
    n : INT, optional
        Logarithm exponent value. The default is 1.
    a : FLOAT, optional
        Polynamial exponent value (can be real). The default is 1.0.

    Returns
    -------
    y : FLOAT or ARRAY (same shape as x)
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
    x : FLOAT or ARRAY
        Input value
    k : FLOAT, optional
        Exponent value, it can theoritically be a real but must
        be an integer if one want to compute the analytical
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
    
def del_u_over_v(du, dv, der) : 
    """
    Computes the derivatives of u/v

    Parameters
    ----------
    du : LIST of FLOAT or ARRAY
        derivatives of u.
    dv : LIST of FLOAT or ARRAY
        derivatives of v.
    der : INT in {0, 1, 2}
        Derivative order.

    Returns
    -------
    y : FLOAT or ARRAY 
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

def interpolate_func(x, y, der=0, k=3, prim_cond=None, *args, **kwargs):
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
    
def lagrange_matrix(x, order=2) :
    """
    Computes the interpolation and derivation matrices based on
    an initial grid x. The new grid, on which the 
    interpolation/derivation takes place, is entierly defined 
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
    mat : ARRAY(N-1, N, 2)
        Contains the interpolation (mat[...,0]) and
        the derivation (mat[...,1]) matrices.

    """
    from init_derive_IFD import init_derive_ifd
    mat_lag, x_lag = init_derive_ifd(x, 1, order)
    return mat_lag


def app_list(val, idx, func=lambda x: x, args=None) :
    """
    Function only designed for convenience in the vectorial mapping finding.

    Parameters
    ----------
    val : LIST
        Values on which func is applied
    idx : LIST
        val ordering.
    func : FUNC or LIST of FUNC, optional
        functions to be applied on val. The default is lambda x: x (identity).
    args : LIST, optional
        list of function arguments. The default is None.

    Returns
    -------
    ARRAY
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
