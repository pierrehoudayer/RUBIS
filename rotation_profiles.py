#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 12:19:57 2022

@author: phoudayer
"""

import numpy as np

from numerical_routines import *

def solid(r, cth, omega) :
    """
    Computes the centrifugal potential and its derivative in the 
    case of a solid rotation profile.

    Parameters
    ----------
    r : float or array_like
        Distance from the origin.
    cth : float
        Value of cos(theta).
    omega : float
        Rotation rate.

    Returns
    -------
    phi_c : float or array_like (same shape as r)
        Centrifugal potential.
    dphi_c : float or array_like (same shape as r)
        Centrifugal potential derivative with respect to r.

    """
    s2, ds2 = r**2 * (1 - cth**2), 2*r * (1 - cth**2)
    phi_c  = -0.5 *  s2 * omega**2
    dphi_c = -0.5 * ds2 * omega**2 
    return phi_c, dphi_c

def lorentzian_profile(r, cth, omega, alpha) :
    """
    Lorentzian rotation profile. The rotation rate difference 
    between the center (c) and the equator (eq) is fixed by 
    alpha = \frac{\Omega_c - \Omega_\mathrm{eq}}{\Omega_\mathrm{eq}}.

    Parameters
    ----------
    r : float or array_like
        Distance from the origin.
    cth : float
        Value of cos(theta).
    omega : float
        Rotation rate on the equator.
    alpha : float
        Rotation rate difference between the center and the equator.

    Returns
    -------
    ws : float or array_like (same shape as r)
        Rotation rate at (r, cth).

    """
    s2 = r**2 * (1 - cth**2)   
    ws = (1 + alpha) / (1 + alpha*s2) * omega
    return ws

def lorentzian(r, cth, omega, alpha) :
    """
    Computes the centrifugal potential and its derivative in the 
    case of a lorentzian rotation profile (cf. function
    lorentzian_profile(r, cth, omega)).

    Parameters
    ----------
    r : float or array_like
        Distance from the origin.
    cth : float
        Value of cos(theta).
    omega : float
        Rotation rate on the equator.
    alpha : float
        Rotation rate difference between the center and the equator.

    Returns
    -------
    phi_c : float or array_like (same shape as r)
        Centrifugal potential.
    dphi_c : float or array_like (same shape as r)
        Centrifugal potential derivative with respect to r.

    """    
    s2, ds2 = r**2 * (1 - cth**2), 2*r * (1 - cth**2)
    phi_c  = -0.5 * s2  * (1 + alpha)**2 / (1 + alpha*s2)**1 * omega**2
    dphi_c = -0.5 * ds2 * (1 + alpha)**2 / (1 + alpha*s2)**2 * omega**2
    return phi_c, dphi_c

def plateau_profile(r, cth, omega, alpha, scale, k=1) : 
    """
    Rotation profile with a "plateau" close to s = 0. It can be used
    to simulate a solid rotation in the center. The rotation rate difference 
    between the center (c) and the equator (eq) is fixed by 
    alpha = \frac{\Omega_c - \Omega_\mathrm{eq}}{\Omega_\mathrm{eq}}. The 
    profile scale can be adjusted using the global parameter 'scale'.

    Parameters
    ----------
    r : float or array_like
        Distance from the origin.
    cth : float
        Value of cos(theta).
    omega : float
        Rotation rate on the equator.
    alpha : float
        Rotation rate difference between the center and the equator.
    scale : float
        Rotation profile scale.
    k : integer, optional
        Value that impacts the plateau length (higher k, smaller plateau). 
        The default is 1.

    Returns
    -------
    ws : float or array_like (same shape as r)
        Rotation rate at (r, cth).

    """    
    corr = np.exp(scale**(2/k))
    w0 = (1 + alpha) * omega
    dw = alpha * omega * corr
    x  = r**2 * (1 - cth**2) / scale**2
    ws = w0 - dw * expinv(x, k)
    return ws

def plateau(r, cth, omega, alpha, scale, k=1) :
    """
    Computes the centrifugal potential and its derivative in the 
    case of a "plateau" rotation profile (cf. function
    plateau_profile(r, cth, omega)).

    Parameters
    ----------
    r : float or array_like
        Distance from the origin.
    cth : float
        Value of cos(theta).
    omega : float
        Rotation rate on the equator.
    alpha : float
        Rotation rate difference between the center and the equator.
    scale : float
        Rotation profile scale.
    k : integer, optional
        Value that impacts the plateau length (higher k, smaller plateau). 
        The default is 1.

    Returns
    -------
    phi_c : float or array_like (same shape as r)
        Centrifugal potential.
    dphi_c : float or array_like (same shape as r)
        Centrifugal potential derivative with respect to r.

    """    
    corr = np.exp(scale**(2/k))
    w0 = (1 + alpha) * omega
    dw = alpha * omega * corr
    s2, ds2 = r**2 * (1 - cth**2), 2*r * (1 - cth**2)
    x = s2 / scale**2
    I1 , I2  = expI(  x, k, 1), expI(  x, k, 2)
    II1, II2 = expinv(x, k, 1), expinv(x, k, 2)
    phi_c  = -0.5 * s2  * (w0**2 - 2*w0*dw * I1  + dw**2 * I2 )
    dphi_c = -0.5 * ds2 * (w0**2 - 2*w0*dw * II1 + dw**2 * II2)
    return phi_c, dphi_c

def la_bidouille(fname, smoothing=0, return_profile=False) : 
    """
    Sets up the function phi_c_func(r, cth, omega) which computes
    the centrifugal potential and its derivative using a numerical 
    rotation profile stored in fname. The interpolation, integration,
    smoothing is handled by a spline decomposition routine. Returns 
    the rotation profile (with same signature) if the option 
    return_profile is set to True.
    
    Parameters
    ----------
    fname : string
        File to be read.
    smoothing : float, optional
        Optional smoothing value applied to the numerical rotation
        profile. 
        The default is 0.
    return_profile : boolean, optional
        Option allowing to return the rotation profile instead of
        the centrifugal potential. The default is False.

    Returns
    -------
    phi_c_func or w_func : function(r, cth, omega)
        cf. below

    """
    dat = np.loadtxt('./Models/'+fname)
    _, idx = np.unique(dat[:, 0], return_index=True)
    sd, wd, _ = dat[idx].T
    
    if return_profile : 
        w_int  = interpolate_func(sd, wd, der=0, s=smoothing)
        dw_int = interpolate_func(sd, wd, der=1, s=smoothing)
        
        def w_func(r, cth, omega) : 
            """
            Compute the rotation profile and its derivative
            based on the imported profile.
            
            Parameters
            ----------
            r : float or array_like
                Distance from the origin.
            cth : float
                Value of cos(theta).
            omega : float
            
                Rotation rate on the equator.

            Returns
            -------
            ws : float or array_like (same shape as r)
                Rotation rate at (r, cth).
            dws : float or array_like (same shape as r)
                Rotation rate derivative with respect to s = r * sth
                /!\ DIFFERENT FROM 'phi_c_func' WHERE THE DERIVATIVE
                IS TAKEN WITH RESPECT TO 'r' (might be a good idea to
                have the same convention though...).
            """
            s = r * (1 - cth**2)**0.5
            corr = (omega/wd[-1])
            ws  =  w_int(s) * corr
            dws = dw_int(s) * corr
            return ws, dws
        
        return w_func
    
    else :     
        phi_c_int  = interpolate_func(sd, -sd * wd**2, der=-1, s=smoothing)
        dphi_c_int = interpolate_func(sd, -sd * wd**2, der= 0, s=smoothing)
        
        def phi_c_func(r, cth, omega) : 
            """
            Computes the centrifugal potential and its derivative 
            using a numerical rotation profile.

            Parameters
            ----------
            r : float or array_like
                Distance from the origin.
            cth : float
                Value of cos(theta).
            omega : float
                Rotation rate on the equator.

            Returns
            -------
            phi_c : float or array_like (same shape as r)
                Centrifugal potential.
            dphi_c : float or array_like (same shape as r)
                Centrifugal potential derivative with respect to r.

            """
            sth = (1 - cth**2)**0.5
            corr = (omega/wd[-1])**2
            phi_c  =  phi_c_int(r * sth) * corr
            dphi_c = dphi_c_int(r * sth) * corr * sth
            return phi_c, dphi_c
        
        return phi_c_func