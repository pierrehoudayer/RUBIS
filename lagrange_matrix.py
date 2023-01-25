#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 16:48:54 2023

@author: phoudayer
"""

import numpy as np
from numpy.polynomial.polynomial import Polynomial
from itertools import combinations

def solve(c):
    
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
    deriv_roots = Polynomial(c_i).roots()
    lb, ub = t_i[order - 1 - max(order - 1 - i, 0)], t_i[order - max(order - 1 - i, 0)]
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
        roots = solve(c.T)
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
    mat_lag, _ = init_derive_ifd(x, 1, order)
    return mat_lag